#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: greenvalley
@file: $NAME.py
@time: 2018/10/11
'''
import cv2
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt

from skimage import measure


def contours_test():
    # Construct some test data
    x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
    r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))

    # Find contours at a constant value of 0.8
    contours = measure.find_contours(r, 0.8)
    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(r, interpolation='nearest', cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def erode_bound():
    z_value = np.array([[0,0,0,0,0], [0, 1, 1, 0, 0], [0, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 1, 0]]).astype('f4')
    element = np.uint8(np.zeros((3, 3)))
    dilate = cv2.dilate(z_value, element)
    pass


if __name__ == '__main__':
    erode_bound()
    pass

