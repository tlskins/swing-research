import numpy as np
import cv2
from IPython.display import Video
from collections import OrderedDict
from scipy.spatial import distance as dist
# %matplotlib inline
from matplotlib import pyplot as plt
from centroid_tracker import CentroidTracker

np.random.seed(42)

INPUT_IMG = 'ball_test.png'
OUTPUT_IMG = 'ball.png'


def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


image = cv2.imread(INPUT_IMG)

# Bound yellow contours
# https://stackoverflow.com/questions/57262974/tracking-yellow-color-object-with-opencv-python
hframe = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([22, 93, 0], dtype="uint8")  # too low
# lower = np.array([22, 150, 150], dtype="uint8") # too high
# lower = np.array([22, 100, 100], dtype="uint8")  # pretty good
# lower = np.array([22, 93, 50], dtype="uint8")  # some noise
upper = np.array([45, 255, 255], dtype="uint8")
mask = cv2.inRange(hframe, lower, upper)

yellowCnts = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
yellowCnts = yellowCnts[0] if len(yellowCnts) == 2 else yellowCnts[1]

yellowRects = []
for cnt in yellowCnts:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w*h
    if area < 75 and area > 70 and w > 0.8*h and w < 1.2*h:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(OUTPUT_IMG, image[y:y+h, x:x+w])
