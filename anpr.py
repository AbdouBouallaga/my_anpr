import cv2
import numpy as np
import imutils
import sys
import os.path
import pytesseract
import numpy as np

dsize = (460, 350)
video = 0

class tabplate:
	def __init__(self):
		self.location = None
		self.approx = None

def fun():
	return tabplate()

if len(sys.argv) != 2:
	print("no or too mush args")
	quit()
if sys.argv[1] == 'cam':
	path = './test.mov'
	video = 1
elif os.path.isfile(sys.argv[1]):
	path = sys.argv[1]
else:
	print("incorrect file")
	quit()

def loadimage(path):
	img = cv2.imread(path)
	return(img)

def grayfilter(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return(gray)

def edgedetection(gray):
	bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
	edged = cv2.Canny(bfilter, 30, 200) #Edge detection
	return(edged)

def findContours(edged):
	keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(keypoints)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
	return(contours)

def compareapprox(approx):
    print(approx[0][0])

def getplate(contours):
	location = None
	found = 0

	ret = tabplate()
	for contour in contours:
		approx = cv2.approxPolyDP(contour, 10, True)
		x, y, w, h = cv2.boundingRect(contour)
		if len(approx) == 4 and w > (h*3):
			location = approx
			x, y, w, h = cv2.boundingRect(contour)
			print('x',x)
			print('y',y)
			print('w',w)
			print('h',h)
            # print(approx[0][0])
			found = 1
			break
	if found != 1:
		print("No plate found exiting")
		return (0)
	ret.approx = approx
	ret.location = location
	return(ret)

def display(title,img):
	img = cv2.resize(img, dsize)
	cv2.imshow(title, img)

def displayn(title,img):
	cv2.imshow(title, img)

def maskandcrop(img, gray, location):
	mask = np.zeros(gray.shape, np.uint8)
	new_image = cv2.drawContours(mask, [location], 0,255, -1)
	new_image = cv2.bitwise_and(img, img, mask=mask)
	(x,y) = np.where(mask==255)
	(x1, y1) = (np.min(x), np.min(y))
	(x2, y2) = (np.max(x), np.max(y))
	cropped = gray[x1:x2+1, y1:y2+1]
	return(cropped)

def gettext(img):
	text = pytesseract.image_to_string(img, lang='ara')
	return (text)


def writeinpic(img, approx, text):
	font = cv2.FONT_HERSHEY_SIMPLEX
	res = cv2.putText(img, text=text, org=(approx[0][0][0]-90, approx[1][0][1]+30), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
	res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
	return (res)

def tiltfix(img):
	# rend = cv2.bitwise_not(img)
	thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	displayn("thresh", thresh)
	coords = np.column_stack(np.where(thresh > 0))
	angle = cv2.minAreaRect(coords)[-1]
	print(angle)
	if angle != 90.0:
		if angle < -45:
			angle = -(90 + angle)
		else:
			angle = -angle
		(h, w) = thresh.shape[:2]
		center = (w // 2, h // 2)
		M = cv2.getRotationMatrix2D(center, angle, 1.0)
		rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
	else:
		rotated = img
	return (rotated)

def process_img(img):
	gray = grayfilter(img)
	display("gray", gray)
	contours = findContours(edgedetection(gray))
	plus = 0
	ret = getplate(contours)
	if ret != 0:
		plate = maskandcrop(img, gray, ret.location)
		displayn("plate", plate)
		plate = tiltfix(plate)
		text = gettext(plate)
		displayn("plate_to_read", plate)
		print(text)
		res = writeinpic(img, ret.approx, text)
		display("result", res)

def runvideo():
	feed = cv2.VideoCapture(0)
	while True:
		_, frame = feed.read()
		if frame is None:
			break
		process_img(frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()

if __name__ == "__main__":
	if video == 1:
		runvideo()
	else:
		img = loadimage(path)
		process_img(img)
	cv2.waitKeyEx()
