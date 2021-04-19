import cv2
import numpy as np
import imutils
import sys
import os.path
import pytesseract

class tabplate:
    def __init__(self):
        self.location = None
        self.approx = None

def fun():
    return tabplate()

if len(sys.argv) != 2:
    print("no or too mush args \n")
    quit()
if os.path.isfile(sys.argv[1]):
    path = sys.argv[1]
else:
    print("incorrect file \n")
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

def findContours(edge):
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    return(contours)

def getplate(plus):
    location = None
    found = 0
    contour = 0
    contour += plus
    ret = tabplate()
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            found = 1
            break
    if found != 1:
        print("No plate found exiting")
        quit()
    ret.approx = approx
    ret.location = location
    return(ret)

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
    text = pytesseract.image_to_string(img)
    return (text)

def writeinpic(img, approx):
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
    return (res)

if __name__ == "__main__":
    img = loadimage(path)
    gray = grayfilter(img)
    cv2.imshow("imagegray",cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    edged = edgedetection(gray)
    contours = findContours(edged)
    plus = 0
    ret = getplate(plus)
    cropped_image = maskandcrop(img, gray, ret.location)
    cv2.imshow("plate",cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    text = gettext(cropped_image)
    res = writeinpic(img, ret.approx)
    cv2.imshow("reading",cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    cv2.waitKeyEx()
