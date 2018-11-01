import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pdf2image
import pytesseract
import pyzbar.pyzbar as pyzbar
import os
from os.path import join
from src.register_image import register_image
from src.utils_uzn import coord2crop

dataroot = 'data'

def process_page(imReg, labels, pageDir):
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
      Image.fromarray(imCrop).save(join(pageDir, labels[(i,j)]+'.jpg'))

# load the blank (template) page
pages = pdf2image.convert_from_path(join(dataroot,'worksheet.pdf'), dpi=300, thread_count=6)
imBlank = np.array(pages[0])

# load the candidate page as image and register it
imOrig = pdf2image.convert_from_path(join(dataroot, 'generated', '111000-111009.pdf'), dpi=300, thread_count=6)
# imOrig = Image.open('worksheet.pdf').resize(imBlank.shape[1::-1])
imOrig = np.array(imOrig[0].resize(imBlank.shape[1::-1]))
imReg, h = register_image(imOrig, imBlank, threshdist=300)

# decode QR code to get page seed
decoded = pyzbar.decode(Image.fromarray(imReg))
assert len(decoded)==1
assert decoded[0].type=='QRCODE'
seed = decoded[0].data.decode('utf-8')

# obtain the labels given the page seed
labels = pickle.load(open(join(dataroot, 'generated', str(seed)+'.pkl'), 'rb'))

# process single page
pageDir = join(dataroot, 'processed', str(seed))
os.makedirs(pageDir, exist_ok=True)
process_page(imReg, labels, pageDir)

