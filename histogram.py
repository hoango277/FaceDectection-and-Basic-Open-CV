import cv2 as cv
import numpy as np

img = cv.imread('Photos/cat.jpg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# Grayscale histogram