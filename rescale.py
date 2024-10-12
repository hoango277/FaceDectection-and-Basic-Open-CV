import cv2

from read import capture
def rescaleFrame(frame, scale = 0.75):
    width = frame.shape[1] * scale
    height = frame.shape[0] * scale

    return cv2.resize(frame, (int(width), int(height)), interpolation = cv2.INTER_AREA)

def changeRes(width, height):
    capture.set(3, width)
    capture.set(4, height)
