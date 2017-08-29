import cv2 #opencv library
import numpy as np
from matplotlib import pyplot as plt
from transform import *
import subprocess
import os
import os.path

# Waiting for the image to get uploaded
while(not os.path.exists("C:\wamp\www\AndroidFileUpload\uploads\document.jpeg")):
	pass

print("Performing OCR")

# Replace with your destination of image
img = cv2.imread('C:\wamp\www\AndroidFileUpload\uploads\document.jpeg')
#img = enhance(img)
org = img

ratio = img.shape[0] / 500.0
img = aspectResize(img, rows = 500)

#visit http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html to know about functions

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#processing image by Blurring, Edge detection and Binarization
img = cv2.GaussianBlur(img, (5, 5), 0)
kernel = np.ones((3,3),np.uint8)
img = cv2.dilate(img,kernel,iterations = 1)
img = cv2.Canny(img, 20, 75)
img = cv2.dilate(img,kernel,iterations = 1)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.show()

#Approximating contours to rectangle to extract out essential part

_, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]


rect = None
for c in contours:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	if len(approx) == 4:
		rect = approx
		break
if(not(rect is None)):
	pts = rect.reshape(4, 2)
	img = four_point_transform(org,pts*ratio)
else:
	img = org

#if document is in landscape mode rotate it
h,w = img.shape[:2]
if(w < h):
	img = rotate_image(img)
org = img

img = process_img(img)	
img = crop_resize(img,org)
img = deskew_image(img)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.show()
cv2.imwrite('final.png',img)

#You change whitelist according to the document to be processed

whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

command = "tesseract final.png out -c tessedit_char_whitelist="+whitelist+" -psm 8"
subprocess.check_call(command)
os.remove("C:\wamp\www\AndroidFileUpload\uploads\document.jpeg")
f = open("out.txt")
print(f.read())
