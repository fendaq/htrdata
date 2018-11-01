import argparse
import pickle
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pdf2image
import pytesseract
import pyzbar.pyzbar as pyzbar
import os
from os.path import join, basename, dirname
from src.utils_uzn import coord2crop
from glob import glob
import re
from itertools import filterfalse
import sys
sys.path.append(join(os.environ['HOME'], 'imreg'))
from register_image import register_image

parser = argparse.ArgumentParser(description='PyTorch MNIST/CIFAR10 Training')
parser.add_argument('--redo', action='store_true', help='force re-proceessing of all files')
args = parser.parse_args()

crowdRoot = 'crowdsource'

def extract_and_save_crops(imReg, labels, saveDir):
  '''process a single page. extract all handwriting and save to file with filename as label'''

  pad = 0
  xAnchor = 421 - pad
  yAnchor = 204 - pad
  xWidth = 460 + 2*pad
  yHeight = 104 + 2*pad
  xCnt = 3
  yCnt = 26
  xSpacing = 770
  ySpacing = 107

  for i in range(xCnt):
    for j in range(yCnt):

      # OPTIONAL: extract the printed form of the number
      coord = [ i * xSpacing + 157, j * ySpacing + 235, 258, 80, 'label' ]
      imCrop, nameCrop = coord2crop(imReg, coord)
      # tessLabel = pytesseract.image_to_string(imCrop, config=('-l eng --oem 1 --psm 3'))
      # print(tessLabel)
      # Image.fromarray(imCrop).show()

      # extract the handwritten number
      coord = [ i * xSpacing + xAnchor, j * ySpacing + yAnchor, xWidth, yHeight, 'image' ]
      imCrop, nameCrop = coord2crop(imReg, coord)
      Image.fromarray(imCrop).save(join(saveDir, labels[(i, j)] + '.jpg'))

def get_seed_from_qr(imReg):
  '''scan the image for QR code. decode it and return the value (page seed)'''
  decoded = pyzbar.decode(Image.fromarray(imReg))
  if not len(decoded)==1:
    warnings.warn(str(len(decoded))+' QR codes found')
    return None
  assert decoded[0].type=='QRCODE'
  seed = decoded[0].data.decode('utf-8')
  return seed

def process_pdf(file, crowdRoot, imBlank):
  '''process a single pdf file'''
  imOrigAll = pdf2image.convert_from_path(join(file), dpi=300, thread_count=6)
  for page, imOrig in enumerate(imOrigAll): # loop over all pages in the pdf

    # register image
    imOrigNp = np.array(imOrig.resize(imBlank.shape[1::-1]))
    imReg, h = register_image(imOrigNp, imBlank, threshdist=300)
    if type(imReg) is not np.ndarray:
      warnings.warn('Image registration failed for '+basename(file)+' page '+str(page))
      continue

    # decode QR
    seed = get_seed_from_qr(imReg)
    if seed==None:
      warnings.warn('Decode QR seed failed for '+basename(file)+' page '+str(page))
      continue
    labels = pickle.load(open(join(crowdRoot, 'generated', 'label-'+str(seed) + '.pkl'), 'rb')) # obtain the labels given the page seed

    # extract and save crops (along with their true labels as the filename) from that page
    saveDir = join(crowdRoot, 'extracted', str(seed))
    os.makedirs(saveDir, exist_ok=True)
    extract_and_save_crops(imReg, labels, saveDir)

# load the blank (template) page
pages = pdf2image.convert_from_path(join(crowdRoot, 'worksheet.pdf'), dpi=300, thread_count=6)
imBlank = np.array(pages[0])

# load the candidate page as image and register it
scribeDir = join(crowdRoot, 'scribed')
files = glob(join(scribeDir, '*.pdf'))
if not args.redo: files = [file for file in files if file.find('extracted-')==-1]

# loop through all pdfs scribed directory
for file in files:
  process_pdf(file, crowdRoot, imBlank)
  os.rename(file, join(dirname(file), 'extracted-'+basename(file))) # add tag to filename so dont waste time to reprocess this file again
