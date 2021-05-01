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
		self.plate = None
		self.text = None

def fun():
	return tabplate()

if len(sys.argv) != 3:
		print("no or too mush args")
		quit()

if sys.argv[1] == 'cam': 
	path = int(sys.argv[2])
	video = 1
elif sys.argv[1] == 'video':
	path = sys.argv[2]
	video = 2
elif sys.argv[1] == 'image':
	path = sys.argv[2]

if not os.path.isfile(sys.argv[2]) and sys.argv[1] != 'cam':
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
	# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
	return(contours)


def getplates(contours):
	found = 0
	location = []

	# ret = tabplate()
	for contour in contours:
		approx = cv2.approxPolyDP(contour, 10, True)
		x, y, w, h = cv2.boundingRect(contour)
		if len(approx) == 4 and w > (h*3):
			location.append(approx)
			x, y, w, h = cv2.boundingRect(contour)
			# print('x',x)
			# print('y',y)
			# print('w',w)
			# print('h',h)
            # print(approx[0][0])
			found = 1
	if found != 1:
		print("No plate found exiting")
		location.append(None)
		return (0)
	# ret.approx = approx
	# ret.location = location
	return(location)

def display(title,img):
	img = cv2.resize(img, dsize)
	cv2.imshow(title, img)

def displayn(title,img):
	cv2.imshow(title, img)
	# print(title)
	# print(img.shape)

def displayloop(title,plates):
	i = 0
	for snap in plates:
		cv2.imshow(title+str(i), snap.plate)
		i += 1
		# print(title)
		# print(snap.plate.shape)

def printloop(texts):
	i = 1
	for text in texts:
		print(i)
		i += 1
		print(text)

def maskandcrop(img, gray, location):
	plates = []
	for plate in location:
		# print(plate)
		mask = np.zeros(gray.shape, np.uint8)
		new_image = cv2.drawContours(mask, [plate], 0,255, -1)
		new_image = cv2.bitwise_and(img, img, mask=mask)
		(x,y) = np.where(mask==255)
		(x1, y1) = (np.min(x), np.min(y))
		(x2, y2) = (np.max(x), np.max(y))
		plates.append(gray[x1:x2+1, y1:y2+1])
	return(plates)

def gettext(imgs):
	ret = []
	for img in imgs:
		text = pytesseract.image_to_string(img)
		ret.append(text)
	return (ret)


def writeinpic(img, approxs, texts):
	font = cv2.FONT_HERSHEY_SIMPLEX
	for text in texts:
		for approx in approxs:
			img = cv2.putText(img, text=text, org=(approx[0][0][0]-90, approx[1][0][1]+30), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
			img = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
	return (img)

def tiltfix(imgs):
	# rend = cv2.bitwise_not(img)
	rotated = []
	for img in imgs:
		thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		displayn("thresh", thresh)
		coords = np.column_stack(np.where(thresh > 0))
		angle = cv2.minAreaRect(coords)[-1]
		# print(angle)
		if angle != 90.0:
			if angle < -45:
				angle = -(90 + angle)
			else:
				angle = -angle
			(h, w) = thresh.shape[:2]
			center = (w // 2, h // 2)
			M = cv2.getRotationMatrix2D(center, angle, 1.0)
			rotated.append(cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)) 
		else:
			rotated.append(img)
	return (rotated)

def	prepare_to_read(imgs):
	# img = cv2.threshold(img, 155, 160, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	size = (80, 15)
	ret = []
	for img in imgs:
		img = cv2.resize(img, size)
		ret.append(img)
	return(ret)

def	verify_content(plates, texts):
	ret = []
	for text, plate in zip(texts, plates):
			res = tabplate()
			res.text = text
			res.plate = plate
			ret.append(res)
	# for r in ret:
		# print(">>>", r.plate)
	return(ret)

def clean(texts, locations):
	news = []
	newl = []
	tex = 0
	for text in texts:
		new = ''
		tex += 1
		leng = 0
		for i in text:
			if (ord(i) >= 47 and ord(i) <= 57) or ord(i) > 1000 or ord(i) == 124:
				# print("this is "+i+" raw =", ord(i))
				new += i
				leng += 1
				# print(">>>>>", new)
		if leng > 2:
			news.append(new)
			x = 0
			for loc in locations:
				x += 1
				if x == tex:
					newl.append(loc)
					break
	# for i in news:
	# 	# print(i)
	# 	for n in i:
	# 		# print(n)
	return (news, newl)

def process_img(img):
	gray = grayfilter(img)
	display("gray", gray)
	contours = findContours(edgedetection(gray))
	# print(contours)
	location = getplates(contours)
	# print(">>>>>>>", location)
	if location != 0:
		plates = maskandcrop(img, gray, location)
		plates = tiltfix(plates)
		plates = prepare_to_read(plates)
		texts = gettext(plates)
		texts, location  = clean(texts, location)
		platesnew = verify_content(plates, texts)
		# displayloop("plate_to_read", platesnew)
		# printloop(texts)
		res = writeinpic(img, location, texts)
		display("result", res)

def runvideo(path):
	print(path)
	feed = cv2.VideoCapture(path)
	while True:
		_, frame = feed.read()
		if frame is None:
			break
		process_img(frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	feed.release()

if __name__ == "__main__":
	if video:
		runvideo(path)
	else:
		img = loadimage(path)
		process_img(img)
	cv2.waitKeyEx()
