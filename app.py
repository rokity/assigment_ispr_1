import matplotlib.pyplot as plt
import sys
import numpy as np
import cv2
import math


def convolution1D(img, kernel, x, y):
    _y = y-1
    pixel = 0
    for i in range(0, 3):
        _x = x-1
        for j in range(0, 3):
            if(_x > -1 and _y > -1 and _x < img.shape[0] and _y < img.shape[1]):
                pix = img[_x, _y]
            else:
                pix = 0
            pixel += (float(kernel[i][j]) * pix)
            _x = _x+1
        _y = _y+1
    return pixel


def sobel_filter(img):

    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel = np.zeros(img.shape)
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            pixel_x = convolution1D(img, kernel_x, x, y)
            pixel_y = convolution1D(img, kernel_y, x, y)
            val = math.sqrt((pixel_x*pixel_x) + (pixel_y*pixel_y))
            sobel[x, y] = val
    return sobel


def main(_file):
    pic = cv2.imread(_file, cv2.IMREAD_GRAYSCALE)
    img = np.matrix(pic)
    img = img/ 255.0
    edge = sobel_filter(img)
    plt.imshow(edge, cmap='gray')
    plt.title('Sobel ')
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1])
