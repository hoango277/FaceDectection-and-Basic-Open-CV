from pickletools import uint8

import numpy as np
import cv2 as cv

blank = np.zeros((500,500,3), dtype='uint8')
# cv.imshow('test',blank)
#
blank[200:300, 300:400] = 0,0,255
cv.imshow('test',blank)

cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=2)
cv.imshow('test', blank)

cv.putText(blank,'hello', (0,255), cv.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2 )
cv.imshow('test', blank)

cv.waitKey(0)