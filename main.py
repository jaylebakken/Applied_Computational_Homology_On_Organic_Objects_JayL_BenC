
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import gudhi
from gudhi.sklearn.rips_persistence import RipsPersistence as rp

def getCloud(image_ndArray):
    points = np.argwhere(image_ndArray<255)
    x = points[:, 1]
    y = points[:, 0]
    return [x,y]
    
def filter(path):
    img = Image.open(path)
    img = img.convert('L')
    imat = np.asarray(img)
    imat = np.where(imat<(np.mean(imat)-np.std(imat)), imat,255)
    #img = Image.fromarray(imat,mode="L")
    #img.save("sampleImages/sampleimg_{}.png".format(int(i)))
    return imat

def showPoints(point_Cloud):
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
def computePD(filtered_image):
    rips = rp([0,1,2], input_type='point cloud', num_collapses='auto', homology_coeff_field=11, n_jobs=None)
    diag = rips.transform([filtered_image])
    return diag

def showPD(persistance_matrix,species):
    ##unwrap PD
    PD = persistance_matrix[0]
    ##
    colors = ['BLUE','RED','GREEN']
    for i in range(0,3):    
        xx = PD[i][:,0]
        yy = PD[i][:,1]    
        plt.scatter(xx,yy, s=5, color=colors[i],label=("dimension " + str(i) + " features"))
    plt.title("Persistance Diagram for " + species)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def main():
    img = filter("Dataset/POLLEN73S/acrocomia_aculeta/Figura2.TIF")
    points = getCloud(img)
    showPoints(points)
    diag = computePD(img)
    showPD(diag, "acrocomia aculeta 1")
    


if __name__=="__main__":
    main()