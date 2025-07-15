from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascades", type=str, default="cascades",
	help="path to input directory containing haar cascades")
args = vars(ap.parse_args())

facexs = []
faceys = []
facexws = []
faceyws = []

# initialize a dictionary that maps the name of the haar cascades to
# their filenames
detectorPaths = {
	"face": "haarcascade_frontalface_default.xml",
	"eyes": "haarcascade_eye.xml",
	"smile": "haarcascade_smile.xml",
	"glasses": "haarcascade_eye_tree_eyeglasses.xml",
	"left": "haarcascade_lefteye_2splits.xml",
	"right": "haarcascade_righteye_2splits.xml",
	"mouth": "haarcascade_mcs_mouth.xml",
}
# initialize a dictionary to store our haar cascade detectors
print("[INFO] loading haar cascades...")
detectors = {}
# loop over our detector paths
for (name, path) in detectorPaths.items():
    # load the haar cascade from disk and store it in the detectors
    # dictionary
    fullPath = os.path.join(args["cascades"], path)
    detectors[name] = cv2.CascadeClassifier(fullPath)
	

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(0).start()
time.sleep(2.0)
#fps = vs.get(cv2.CAP_PROP_FPS)
fps = 30
frame_count = 0
print_time = 1
frame_center_x = 250
deviations = []

# loop over the frames from the video stream
while True:
	# grab the frame from the video stream, resize it, and convert it
	# to grayscale
	frame = vs.read()
	#frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# perform face detection using the appropriate haar cascade
	faceRects = detectors["face"].detectMultiScale(
		gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
	for (fX, fY, fW, fH) in faceRects:
		face_center_x = fX + fW / 2
		deviation = face_center_x - frame_center_x
		deviations.append(deviation)	
	
    # loop over the face bounding boxes
	for (fX, fY, fW, fH) in faceRects:
		facexs.append(fX)
		faceys.append(fY)
		facexws.append(fW)
		faceyws.append(fH)


		# extract the face ROI
		faceROI = gray[fY:fY+ fH, fX:fX + fW]
		# apply eyes detection to the face ROI
		eyeRects = detectors["eyes"].detectMultiScale(
			faceROI, scaleFactor=1.1, minNeighbors=10,
			minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
		# apply smile detection to the face ROI
		smileRects = detectors["smile"].detectMultiScale(
			faceROI, scaleFactor=1.1, minNeighbors=10,
			minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
		
		leftRects = detectors["left"].detectMultiScale(
			faceROI, scaleFactor=1.1, minNeighbors=10,
			minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
		
		rightRects = detectors["right"].detectMultiScale(
			faceROI, scaleFactor=1.1, minNeighbors=10,
			minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
		
		glassesRects = detectors["glasses"].detectMultiScale(
			faceROI, scaleFactor=1.1, minNeighbors=10,
			minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
		
		mouthRects = detectors["mouth"].detectMultiScale(
			faceROI, scaleFactor=1.1, minNeighbors=10,
			minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
		

		for (eX, eY, eW, eH) in eyeRects:
			# draw the eye bounding box
			ptA = (fX + eX, fY + eY)
			ptB = (fX + eX + eW, fY + eY + eH)
			cv2.rectangle(frame, ptA, ptB, (0, 0, 255), 2)

		# loop over the smile bounding boxes
		for (sX, sY, sW, sH) in smileRects:
			# draw the smile bounding box
			ptA = (fX + sX, fY + sY)
			ptB = (fX + sX + sW, fY + sY + sH)
			cv2.rectangle(frame, ptA, ptB, (255, 0, 0), 2)

		for (lX, lY, lW, lH) in leftRects:
			# draw the smile bounding box
			ptA = (fX + lX, fY + lY)
			ptB = (fX + lX + lW, fY + lY + lH)
			cv2.rectangle(frame, ptA, ptB, (0, 255, 0), 2)

		for (rX, rY, rW, rH) in rightRects:
			# draw the smile bounding box
			ptA = (fX + rX, fY + rY)
			ptB = (fX + rX + rW, fY + rY + rH)
			cv2.rectangle(frame, ptA, ptB, (60, 60, 0), 2)

		for (mX, mY, mW, mH) in mouthRects:
			# draw the smile bounding box
			ptA = (fX + mX, fY + mY)
			ptB = (fX + mX + mW, fY + mY + mH)
			cv2.rectangle(frame, ptA, ptB, (60, 0, 60), 2)

		for (gX, gY, gW, gH) in glassesRects:
			# draw the smile bounding box
			ptA = (fX + gX, fY + gY)
			ptB = (fX + gX + gW, fY + gY + gH)
			cv2.rectangle(frame, ptA, ptB, (0, 60, 60), 2)


		# draw the face bounding box on the frame
		cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
			(0, 255, 0), 2)
		
        	# show the output frame
		
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

timestamps = range(len(deviations))

data = np.array([facexs,faceys,facexws,faceyws])
xaverage = np.average(np.array(facexs))
yaverage = np.average(np.array(faceys))
xwaverage = np.average(np.array(facexws))
ywaverage = np.average(np.array(faceyws))
xstddev = np.std(np.array(facexs))
ystddev = np.std(np.array(faceys))

print("xaverage:")
print(xaverage)
print("yaverage:")
print(yaverage)
print("xwidthaverage:")
print(xwaverage)
print("ywidthaverage:")
print(ywaverage)
print("xstddev:")
print(xstddev)
print("ystddev:")
print(ystddev)

plt.figure(figsize=(10, 6))
plt.plot(timestamps, deviations, label='Head Deviation from Center')
plt.xlabel('Frame')
plt.ylabel('Deviation (pixels)')
plt.legend()
plt.show()

df = pd.DataFrame({'Timestamp': timestamps, 'Deviation': deviations})
df.to_csv('head_deviation.csv', index=False)
