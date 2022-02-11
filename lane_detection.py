# very simple lane detection program which is 
# highly inefficient, uses Hough transform 

# importing libs
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

# here I'll define different methods required for this

def regionOfInterest(image):

    height = image.shape[0]
    width = image.shape[1]

    #all numpy images are arrays, shape[0] is m and 1 is n. mxn image

    polygons = np.array([[(int(0), height), (int(width/2.5), int(0)), (int(width/1.9), int(0)), (int(width), int(0))]])

    # here, This step is to take into account only the region covered by the road lane. 
    # A mask is created here, which is of the same dimension as our road image. 
    # Furthermore, bitwise AND operation is performed between each pixel of our canny image and this mask. 
    # It ultimately masks the canny image and shows the region of interest traced by the polygonal contour of the mask.

    # creating a zero array size of img with np function
    zeroMask = np.zeros_like(image)

    #fill polygon with ones with cv function, we give it array, which points to fill and the val
    cv2.fillConvexPoly(zeroMask, polygons, 1)

    #masking original image to create roi
    roi = cv2.bitwise_and(image, image, mask = zeroMask)

    return roi

def getLineCoordinates(frame, lines):
    slope, yIntercept = lines[0], lines[1]

    # get y and x coordinates

    y1 = frame.shape[0]
    y2 = int(y1-250)

    x1 = int((y1-yIntercept)/slope)
    x2 = int((y2-yIntercept)/slope)

    return np.array([x1,y1,x2,y2])

    #average out lines from hough transformation

def getLines(frame, lines):
    copyImage = frame.copy()
    leftLine, rightLine = [], []
    lineFrame = np.zeros_like(frame)
    for line in lines: 
        x1, y1, x2, y2 = line[0]
        # calculate slope & y intercept
        lineData = np.polyfit((x1,x2), (y1,y2), 1)
        slope, yIntercept = round(lineData[0], 1), lineData[1]
        if slope < 0: 
            leftLine.append((slope, yIntercept))
        else:
            rightLine.append((slope, yIntercept))

    # convert back into [x1,y1,x2,y2] format
    if leftLine:
        leftLineAverage = np.average(leftLine, axis = 0)
        left = getLineCoordinates(frame, leftLineAverage)
        try:
            cv2.line(lineFrame, (left[0],left[1]), (left[2],left[3]), (255, 0, 0), 2)
        except Exception as e: 
            print('Error', e)
    if rightLine:
        rightLineAverage = np.average(rightLine, axis = 0) 
        right = getLineCoordinates(frame, rightLineAverage)
        try:
            cv2.line(lineFrame, (right[0],right[1]), (right[2],right[3]), (255, 0, 0), 2)
        except Exception as e: 
            print('Error:', e)

    # return copyImage with lines weighted for transparency
    return cv2.addWeighted(copyImage, 0.8, lineFrame, 0.8, 0.0)

# save video 

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('ouput.mp4', fourcc, 20.0, (640,360))

vid = cv2.VideoCapture('lane_vgt.mp4')
count = 0
while (vid.isOpened()):
    ret,frame = vid.read()
    cv2.imshow("original frame", frame)

    # 1. convert to grayscale
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. gaussian blur the grayscale frame to reduce noise, kernal size must be positive & odd
    kernel_size = 1
    gaussianFrame = cv2.GaussianBlur(grayFrame,(kernel_size,kernel_size),kernel_size)
    cv2.imshow('gaussianFrame', gaussianFrame)

    # 3. detect edges using canny edge detection
    low_threshold = 45
    high_threshold = 260
    edgeFrame = cv2.Canny(gaussianFrame, low_threshold, high_threshold)

    # 4. define region of interest
    roiFrame = regionOfInterest(edgeFrame)

    # 5. detect lines using hough's algorithm
    lines = cv2.HoughLinesP(roiFrame, 1, np.pi/180, 30, maxLineGap=300)

    imageWithLines = getLines(frame, lines)
    cv2.imshow("final", imageWithLines)



    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

vid.release()
out.release()
cv2.destroyAllWindows()
