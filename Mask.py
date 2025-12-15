import cv2
import matplotlib.pyplot as plt
import os

import numpy as np
import cv2


def getCloud(image_ndArray):
    ##parameters, 2d Nd Array representing an image
    #epsilon, filtration value for the point cloud, default=100
    ##returns an array [x,y] representing a point cloud
    points = np.argwhere(image_ndArray > (np.mean(image_ndArray)+0.85*np.std(image_ndArray)))
    x = points[:, 1]
    y = points[:, 0]
    return [x, y]

def showPoints(point_Cloud):
    # Parameter: [x,y] ndarray representing point cloud
    # Plots point cloud
    x = point_Cloud[0]
    y = point_Cloud[1]
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=1, color='black')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.title(" Edge Detection Pixels Point Cloud")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def showPollen(crop=20):
    path = "Dataset/"
    dataset = os.listdir(path)
    #parse data
    for folder in dataset:
        currentfolder = os.listdir(path+folder + "/")
        for species in currentfolder:
            sample = os.listdir(path + folder + "/" + species + "/")
            img = cv2.imread(path + folder + "/" + species + "/"+sample[0])
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            #show unfiltered image
            cv2.imshow("img", img)
            #boost contrast with clahe
            img = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            cv2.waitKey(0)
            # show filtered image
            mask = fractional_edge_dft(img, 1)
            cv2.imshow("edge_detect", mask[0])
            cv2.waitKey(0)
            # show fractional filtered image
            mask = fractional_edge_dft(img, 0.75)
            cv2.imshow("edge_detect", mask[0])
            cv2.waitKey(0)
            showPoints(getCloud(mask[0][crop:len(mask[0][1:])-crop,crop:len(mask[0][:1])-crop]))

            cv2.waitKey(0)




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

def camStart():
    """Video feed from laptop camera, camera handling"""
    cam = cv2.VideoCapture(0)
    trackWindow1 = None
    trackWindow2 = None

    while True:
        ret, frame = cam.read()
        if not ret:
            print("No more frames...")
            break

        frame = cv2.flip(frame, 1)
        hgt, wid, dep = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge = fractional_edge_dft(gray, alpha=1)

        cv2.imshow('Edge_Detection, 0.5', edge[0])
        v = cv2.waitKey(5)
        if v > 0 and chr(v) == 'q':
            break


if __name__ == '__main__':
    camStart()
    showPollen()
