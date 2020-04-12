import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse

from PIL import Image

def main(k,img): 
    for i, row in enumerate(img):
        for j, col in enumerate(row):
            if row[j]*255 >= k:
                row[j] = 255
            else:
                row[j] = 0
    
    newPxl = np.asarray(img).astype(np.uint32)  
    newImage = Image.fromarray(newPxl*255)
    
    return newImage

def convert(img):
    grayScaleImg = np.zeros(img.shape)
    for i, row in enumerate(img):
        for j, col in enumerate(row):
            pixel = img[i,j,:]
            grayScaleImg[i,j] = 255 * (pixel[0] + pixel[1] + pixel[2]) / 3

    grayScale = np.asarray(grayScaleImg).astype(np.uint32)  
    image = Image.fromarray(grayScale)
    
    return grayScale

if __name__ == '__main__':
    inImage = sys.argv[1]
    outImage = sys.argv[2]
    k = int(sys.argv[3])
    img = plt.imread(inImage)
    if len(img.shape) == 3:
        img = convert(img)
    newImage = main(k,img)
    #newImage.show()
    newImage.save(outImage)
