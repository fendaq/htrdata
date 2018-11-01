import numpy as np
import cv2
from os.path import basename, join
import os
import sys
import matplotlib.pyplot as plt

def put_patch(im, patchPath, centroid):
  '''
  Takes patch image located in patchPath and superimpose on image (im) at centroid. Returns modified image.
  :param im: numpy matrix of greyscale image on which to impose patch
  :param patchPath: path of the patch image (underline, box, spurious text, etc
  :param centroid: centroid of the numpy matrix of image on which to align centroid of patch
  :return: modified (patched) image
  '''

  return imMod


def superimpose_image(imBase, imArt, expand=False):
  '''
  Superimpose artifacts from imArt onto imBase
  :param imBase: master image on which imArt is imposed. whitespace on this image are kept
  :param imArt: image to superimpose. whitespace on this image (pixels below threshold) rendered transparent
  :param expand: True if allow resizing of imMaster
  :return: modified image
  '''

  return imMasterMod



