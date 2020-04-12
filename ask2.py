import sys
import matplotlib.pyplot as plt
import numpy as np
import string

from PIL import Image


def main(img,T,Tt):
    T = np.matmul(Tt, T)
    Tinv = np.linalg.inv(T) 
    newImg = nnInterpolation(Tinv,img)
    newImg = np.asarray(newImg)#.astype(np.uint8) 
    newImage = Image.fromarray(newImg*255)
    
    return newImage

def nnInterpolation(Tinv,img):
    newImg = np.zeros(img.shape)
    xMax, yMax = img.shape[0] - 1, img.shape[1] - 1
    for i, row in enumerate(img):
        for j, col in enumerate(row):
            x, y, _ = np.matmul(Tinv, np.array([i, j, 1]))
            if np.floor(x) == x and np.floor(y) == y:
                newImg[i, j] = img[int(x), int(y)]
                continue
            if np.abs(np.floor(x) - x) < np.abs(np.ceil(x) - x):
                x = int(np.floor(x))
            else:
                x = int(np.ceil(x))
            if np.abs(np.floor(y) - y) < np.abs(np.ceil(y) - y):
                y = int(np.floor(y))
            else:
                y = int(np.ceil(y))
            if x > xMax:
                x = xMax
            if y > yMax:
                y = yMax
            newImg[i, j] = img[x, y]
    
    return newImg
            
if __name__ == '__main__':
    inImage = sys.argv[1]
    outImage = sys.argv[2]
    img = plt.imread(inImage)
    Tt = np.array([[1, 0, -np.round(img.shape[0]/2)], [0, 1, -np.round(img.shape[1]/2)], [0, 0, 1]])
    T = np.array([[float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])], [float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8])], [0, 0, 1]])
    newImage = main(img,T,Tt)
    newImage.show()
    newImage.save(outImage)
    

    