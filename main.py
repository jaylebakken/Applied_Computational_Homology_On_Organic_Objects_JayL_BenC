
import os

import cv2
from PIL import Image
import gudhi.bottleneck
import gudhi.hera
import matplotlib.pyplot as plt
import numpy as np
import gudhi
from gudhi.sklearn.rips_persistence import RipsPersistence as rp, RipsPersistence
import sklearn
import RipsAnimation


    
def filter(path):
    #input: path of image
    #returns a filtered version of image
    img = Image.open(path)
    #Black and white conversion
    img = img.convert('L')
    imat = np.asarray(img)
    #filter image to points outside of standard deviation of image contrast
    imat = np.where(imat<(np.mean(imat)-np.std(imat)), imat,255) 
    #Optional image saving:
    #img = Image.fromarray(imat,mode="L")
    #img.save("sampleImages/sampleimg_{}.png".format(int(i)))
    return imat

def showPoints(point_Cloud):
    #Parameter: [x,y] ndarray representing point cloud
    #Plots point cloud
    x = point_Cloud[0]
    y = point_Cloud[1]
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=1, color='black') 
    plt.gca().invert_yaxis() 
    plt.axis('equal')
    plt.title("Nonwhite Pixels Point Cloud")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def computePD(filtered_image, dimensions=[0,1,2], input_type_='point cloud', homo_field=11):
    rips = rp(homology_dimensions=dimensions, input_type= input_type_, num_collapses='auto', homology_coeff_field=homo_field, n_jobs=None)
    diag = rips.transform([filtered_image])
    return diag

def showPD(persistance_matrix,species):
    ##Param: persistance_matrix as 2d ndarray, species as string
    ##Plots persistance diagram
    ##unwrap PD
    PD = persistance_matrix[0]
    ##
    colors = ['BLUE','RED','GREEN'] 
    for i in range(0,3):    
        xx = PD[i][:,0]
        yy = PD[i][:,1]    
        plt.scatter(xx,yy, s=5, color=colors[i],label=("dimension " + str(i) + " features"))
    plt.title("Persistance Diagram for " + species)
    plt.xlabel("Feature Birth")
    plt.ylabel("Feature Death")
    plt.legend()
    plt.show()

def findMatch(image_path):
    #param: image path
    #returns: closest species match
    query_img = filter(image_path)
    query_points = getCloud(query_img)
    showPoints(query_points)
    query_PD = computePD(query_img)

    path = "Preprocessed_Data/"
    dataset = os.listdir(path)
    closesetmatch = ""
    min = 10000000000000000
    for folder in dataset:
        bottleneck=0
        average = 0
        current_species = os.listdir(path+folder+"/")
        for sample in current_species:
           dimension = 0
           for matrix in os.listdir(path+folder+"/"+sample):
               print(path+folder+"/"+sample+"/"+matrix)
               targetPD= np.load(path+folder+"/"+sample+"/"+matrix)
               bottleneck+= gudhi.hera.bottleneck_distance(targetPD,query_PD[0][dimension])
               dimension +=1
               average+=1
        print(bottleneck/average)
        if bottleneck/average<min:
            min=bottleneck/average
            closesetmatch = folder

    return closesetmatch

def fractional_edge_dft(gray, alpha=0.7):

    f = gray.astype(np.float32)
    F = np.fft.fft2(f)
    Fshift = np.fft.fftshift(F)

    #generate frequency grids in freq. dom
    rows, cols = gray.shape
    u = np.linspace(-0.5, 0.5, cols, endpoint=False)
    v = np.linspace(-0.5, 0.5, rows, endpoint=False)
    U, V = np.meshgrid(u, v)

    #Compute Differentiation mask (freq interval of 2pi)
    kx = 2 * np.pi * U
    ky = 2 * np.pi * V
    epsilon = 1e-12
    #FTT fractional derivative based on Tseng, Pei & Hsia, Signal Processing, 80(1), 151–159, 2000
    # (i*angular frequency)^alpha
    Dx = (1j * (kx))**alpha
    Dy = (1j * (ky))**alpha

    # Apply fractional differentiation masks
    Fx = Fshift * Dx
    Fy = Fshift * Dy
    fx = np.fft.ifft2(np.fft.ifftshift(Fx))
    fy = np.fft.ifft2(np.fft.ifftshift(Fy))

    #Classic edge definition based on dist. formula
    mag = np.sqrt(np.abs(fx)**2 + np.abs(fy)**2)
    # Normalize to displayable 0–255
    mag = (255 * mag / mag.max()).astype(np.uint8)
    return (mag, U,V)

def getCloud(image_ndArray):
        ##parameters, 2d Nd Array representing an image
        # epsilon, filtration value for the point cloud, default=100
        ##returns an array [x,y] representing a point cloud
        points = np.argwhere(image_ndArray > (np.mean(image_ndArray) + 0.85 * np.std(image_ndArray)))
        x = points[:, 1]
        y = points[:, 0]
        return [x, y]

def crop(image_ndArray,width=2,ratio=20):
    '''
    :param image_ndArray:
    :return: cropped image
    '''
    cropX = (len(image_ndArray[1:])//ratio)+width
    cropY = (len(image_ndArray[:1])//ratio)+width

    cropped_img = image_ndArray[cropX:len(image_ndArray[1:]) - cropX, cropY:len(image_ndArray[:1]) - cropY]
    return cropped_img

def wrap_xy(xy_lists):
    """
    Convert [[x1, x2, ...], [y1, y2, ...]] → [[x1, y1], [x2, y2], ...]
    """
    x_values, y_values = xy_lists
    points = np.column_stack((x_values, y_values))
    return points
            
            
def main():
    image_path = "Dataset/POLLEN73S/piper_aduncum/piper_aduncum (4).tif"
    query_img = cv2.imread(image_path)

    #image contrast change
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    query_img = clahe.apply(cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY))
    mask = fractional_edge_dft(query_img, 0.75)
    cv2.imshow("edge_detect", mask[0])
    cv2.waitKey(0)
    cloud = getCloud(crop(mask[0]))
    cloud = cloud / np.max(cloud)
    showPoints(cloud)
    cloud = wrap_xy(cloud)

    animator = RipsAnimation.RipsFiltrationAnimator(
        cloud,
        max_edge_length=1.5,
        max_dimension=2
    )
    animator.save_frames()
    animator.save_gif("piperaduncum.gif")




    #animator.save_frames()
    #animator.save_gif("rips.gif")

    


if __name__=="__main__":
    main()