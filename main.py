
import os
from PIL import Image
import gudhi.bottleneck
import gudhi.hera
import matplotlib.pyplot as plt
import numpy as np
import gudhi
from gudhi.sklearn.rips_persistence import RipsPersistence as rp

def getCloud(image_ndArray):
    ##parameters, 2d Nd Array representing an image
    ##returns an array [x,y] representing a point cloud
    points = np.argwhere(image_ndArray<255)
    x = points[:, 1]
    y = points[:, 0]
    return [x,y]
    
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
    #query_points = getCloud(query_img)
    #showPoints(query_points)
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
           
               
            
            
def main():
    print(findMatch("Dataset/POLLEN73S/anadenanthera_colubrina/anadenanthera_colubrina (3).jpg"))
    


if __name__=="__main__":
    main()