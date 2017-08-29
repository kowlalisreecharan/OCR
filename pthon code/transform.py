import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
	
def auto_canny(img):
	ave_row = np.average(img, axis=0)
	ave = np.average(ave_row, axis=0)
	canny = cv2.Canny(img, ave - 100, ave)
	return canny


def rotate_image(img, angle = 90):
	
	height, width = img.shape[:2]
	image_center = (width / 2, height / 2)

	rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

	radians = math.radians(angle)
	sin = math.sin(radians)
	cos = math.cos(radians)
	bound_w = int((height * abs(sin)) + (width * abs(cos)))
	bound_h = int((height * abs(cos)) + (width * abs(sin)))

	rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
	rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

	rotated_mat = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))
	return rotated_mat
	
#Resize image to make it compatible------------------------------------------------------------------------------- 

def aspectResize(img,rows=None,cols=None):
	
	if rows==None and cols ==None:
		return img
	if rows==None:
		ratio=float(img.shape[1])/cols
	if cols==None:
		ratio=float(img.shape[0])/rows
	
	img=cv2.resize(img,(int(img.shape[1]/ratio),int(img.shape[0]/ratio)))
	return img
	
#Find block of text by using its approximate location and minimum bounding rectangle-----------------------------
#these fields are specific to PAN card find more in documentation

def crop_resize(img,org):
	#img = img[1500:1675 , 0:1250]
	#img = cv2.resize(img,(1250 , 175), interpolation = cv2.INTER_CUBIC)
	
	img = cv2.bitwise_not(img)
	kernel = np.ones((3,3), np.uint8)
	img = cv2.erode(img, kernel, iterations = 1)
	kernel = np.ones((21,21),np.uint8)
	opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	img = cv2.medianBlur(img,15)
	img = cv2.bitwise_not(img)
	img = cv2.resize(img, (695, 519), interpolation=cv2.INTER_CUBIC)
	org = cv2.resize(org, (695, 519), interpolation=cv2.INTER_CUBIC)
	plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
	plt.show()
	
	
	canny = cv2.Canny(img, 100, 200)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
	close = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
	dilate = cv2.dilate(close, kernel, iterations=1)
	#parametres to crop pan card number and BITS-ID card create similar arrays for required fields
	pan = [250 , 340 , 400, 10 ,5 , 0]
	BIC = [600 , 150 , 225 ,10 ,5 ,175]
	
	var = pan
	_, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		if h > dilate.shape[0]*0.025 and h < dilate.shape[0]*0.06 :
			if(x+w/2 < var[0] and y+h/2 > var[1] and y+h/2 < var[2] and w/h < var[3] and w/h > var[4] and x+w/2 > var[5]):
				img = img[y-h/10:y+h/10+h , x-w/10:x+w/10+w]
	img = cv2.resize(img , (2048 , 128))
	return img

#Process the image---------------------------------------------------------------------------------------------

def process_img(img):
	plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
	plt.show()
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img,(830 , 520))
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3)
	img = aspectResize(img,rows = 2048)
	img = cv2.GaussianBlur(img, (3, 3), 0)
	return img

#Identify points of rectangle ----------------------------------------------------------------------------------

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

#Perspective transform the image --------------------------------------------------------------------------
#points are identified and perspective transformed to a size equal to maximum of length of opposite sides

	
def four_point_transform(image, pts):

	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))


	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))


	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")


	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))


	return warped

#de-skew text if skewed -----------------------------------------------------------------------------------
	
def deskew_image(img):
	org = img
	img = cv2.bitwise_not(img)
	coords = np.column_stack(np.where(img > 0))
	angle = cv2.minAreaRect(coords)[-1]
	if angle < -45:
		angle = -(90 + angle)
	else:
		angle = -angle
	(h, w) = img.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	img = cv2.warpAffine(org, M, (w, h),
		flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	print(angle)
	return img