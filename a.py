import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread("./data/Images/Mountain/2.jpg")
img2 = cv2.imread("./data/Images/Mountain/3.jpg")
img3 = cv2.imread("./data/Images/Field/3.jpg")
img4 = cv2.imread("./data/Images/Field/4.jpg")
img5 = cv2.imread("./data/Images/Field/5.jpg")
img6 = cv2.imread("./data/Images/Field/6.jpg")

print(img1)
cv2.imshow("img1",img1)
cv2.waitKey(0)
# find correlation between each of the images

gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img1[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('dst',img1)
cv2.waitKey(0)

# sift = cv2.SIFT_create()
# kp = sift.detect(gray,None)
# img1=cv2.drawKeypoints(gray,kp,img1)
# cv2.imshow('dst',img1)
# cv2.waitKey(0)


# Histogram Analysis:

# Histogram Spread: Analyze the histogram of pixel intensities. A wider spread indicates a greater range of intensities, which may imply the presence of distinct features.
# Peak Detection: Look for peaks in the histogram, as they may correspond to distinct features or objects.

# Calculate the histogram
histogram = cv2.calcHist([img2], [0], None, [256], [0, 256])

# Plot the histogram
plt.plot(histogram)
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
