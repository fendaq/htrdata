from glob import glob
from os.path import join
import numpy as np
import cv2
import matplotlib.pyplot as plt

def register_image(im1, im2, h=None, maxmatch=30000, percentkeep=0.15, threshdist=125):
  '''im2 and im1 are grayscale images (numpy matrices). Align im1 to im2.
  im1 should be the candidate image and im2 should be the template image.
  Returns transformed (aligned) version of im1, also returns homography matrix'''

  if h is not None: # if you only want to apply an existing homography
    height, width, channels = im2.shape
    return cv2.warpPerspective(im1, h, (width, height)), h

  # convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  # detect ORB features and compute descriptors.
  # orb = cv2.xfeatures2d.SIFT_create(maxmatch) # unfortunately jcould not use sift algo unless we pay
  orb = cv2.ORB_create(maxmatch) # we'll settle on the free orb feature detector
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

  # match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # remove not so good matches
  numGoodMatches = int(len(matches) * percentkeep)
  matches = matches[:numGoodMatches]
  # matches, keypoints1, keypoints2 = map(lambda x: x[:numGoodMatches], [matches, keypoints1, keypoints2])

  # extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # compute pairwise distance between matched points, and filter out those that exceed a distance threshold
  distance = np.linalg.norm(points1-points2, axis=1)
  points1 = points1[distance<threshdist]
  points2 = points2[distance<threshdist]
  matches = [match for dist,match in zip(distance,matches) if dist<threshdist]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("feature-matches.jpg", imMatches)

  # plot the pairwise distances
  # plt.hist(distance, 300); plt.xlabel('keypoint pairwise distance'); plt.title('Pairwise distances')
  # plt.show()

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))

  return im1Reg, h
