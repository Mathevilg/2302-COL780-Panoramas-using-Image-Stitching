import numpy as np
import cv2
import matplotlib.pyplot as plt
# # read images
# img01 = cv2.imread('./data/Images/Mountain/1.jpg')  
# img02 = cv2.imread('./data/Images/Mountain/2.jpg') 

# img1 = cv2.cvtColor(img01,cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img02, cv2.COLOR_BGR2GRAY)

# #sift
# sift = cv2.xfeatures2d.SIFT_create()

# keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
# keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

# #feature matching
# bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

# matches = bf.match(descriptors_1,descriptors_2)
# matches = sorted(matches, key = lambda x:x.distance)

# img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
# plt.imshow(img3),plt.show()


# img1 = cv2.imread('./data/Images/Mountain/1.jpg',cv2.IMREAD_GRAYSCALE)          # queryImage
# img2 = cv2.imread('./data/Images/Mountain/2.jpg',cv2.IMREAD_GRAYSCALE) # trainImage
# # Initiate ORB detector
# orb = cv2.ORB_create()
# # find the keypoints and descriptors with ORB
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)

# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# # Match descriptors.
# matches = bf.match(des1,des2)
# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)
# # Draw first 10 matches.
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()

# import numpy as np
# import cv2 as cv
# import matplotlib.pyplot as plt
# img1 = cv.imread('./data/Images/Mountain/2.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
# img2 = cv.imread('./data/Images/Mountain/3.jpg',cv.IMREAD_GRAYSCALE) # trainImage
# # Initiate SIFT detector
# sift = cv.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# # BFMatcher with default params
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1,des2,k=2)
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
# # cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3),plt.show()


image1 = cv2.imread('./data/Images/Office/3.jpg',cv2.COLOR_BGR2GRAY)  
image2 = cv2.imread('./data/Images/Office/5.jpg',cv2.COLOR_BGR2GRAY)  

# Perform feature detection and matching (SIFT)
sift = cv2.SIFT_create()

key_points1, descriptors1 = sift.detectAndCompute(image1, None)
key_points2, descriptors2 = sift.detectAndCompute(image2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Draw matches
print(key_points1)
print(key_points2)
print(good_matches)
matching_result = cv2.drawMatches(image1, key_points1, image2, key_points2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display matches
cv2.imshow('Matches', matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
