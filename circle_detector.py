import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# load the image
img = cv2.imread('ball.jpg')
# convert BGR to RGB to be suitable for showing using matplotlib library
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# make a copy of the original image
cimg = img.copy()
# convert image to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# apply a blur using the median filter
img = cv2.medianBlur(img, 5)
# finds the circles in the grayscale image using the Hough transform
circles = cv2.HoughCircles(image=img, method=cv2.HOUGH_GRADIENT, dp=0.9,
                           minDist=80, param1=110, param2=39, minRadius=1, maxRadius=70)

for co, i in enumerate(circles[0, :], start=1):
    print(i)
    # draw the outer circle
    cv2.circle(cimg, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(cimg, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)

# print the number of circles detected
print("Number of circles detected:", co)
# save the image, convert to BGR to save with proper colors
# cv2.imwrite("coins_circles_detected.png", cimg)
# show the image
plt.imshow(cimg)
plt.show()
