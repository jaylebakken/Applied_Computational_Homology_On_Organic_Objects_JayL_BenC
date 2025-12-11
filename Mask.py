import cv2
import matplotlib.pyplot as plt
import os

import numpy as np
import cv2


def showPollen():
    path = "Dataset/"
    dataset = os.listdir(path)
    for folder in dataset:
        currentfolder = os.listdir(path+folder + "/")
        for species in currentfolder:
            sample = os.listdir(path + folder + "/" + species + "/")
            img = cv2.imread(path + folder + "/" + species + "/"+sample[0])
            cv2.imshow("img", img)
            cv2.waitKey(0)
            mask = fractional_edge_dft(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0.75)
            cv2.imshow("edge_detect", mask[0])
            cv2.waitKey(0)





def fractional_edge_dft(gray, alpha=0.7):

    f = gray.astype(np.float32)

    #Compute fft
    F = np.fft.fft2(f)
    #Center data
    Fshift = np.fft.fftshift(F)

    #generate frequency grids
    rows, cols = gray.shape
    u = np.linspace(-0.5, 0.5, cols, endpoint=False)
    v = np.linspace(-0.5, 0.5, rows, endpoint=False)
    U, V = np.meshgrid(u, v)

    #Compute Differentiation masks
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

    # 6. Inverse FFT
    fx = np.fft.ifft2(np.fft.ifftshift(Fx))
    fy = np.fft.ifft2(np.fft.ifftshift(Fy))

    #Compute magnitude via sum of squares

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
        edge = fractional_edge_dft(gray, alpha=0.7)

        cv2.imshow('Edge_Detection, 0.5', edge[0])
        v = cv2.waitKey(5)
        if v > 0 and chr(v) == 'q':
            break


if __name__ == '__main__':
    camStart()
    showPollen()
