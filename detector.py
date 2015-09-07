import os
import sys

import cv2
import dlib
import numpy as np


class Detector:
	def detect(self, src):
		raise NotImplementedError("Every Detector must implement the detect method.")


class CascadedDetector(Detector):
	def __init__(self):
		pass
	
	def detect(self, src):
		if np.ndim(src) == 3:
			src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
		src = cv2.equalizeHist(src)
		detector = dlib.get_frontal_face_detector()
		rects = detector(src, 1)
		if len(rects) == 0:
			return []
		return rects

		
if __name__ == "__main__":
	# script parameters
	if len(sys.argv) < 2:
		raise Exception("No image given.")
	inFileName = sys.argv[1]
	outFileName = None
	if len(sys.argv) > 2:
		outFileName = sys.argv[2]
	if outFileName == inFileName:
		outFileName = None
	# detection begins here
	img = np.array(cv2.imread(inFileName), dtype=np.uint8)
	imgOut = img.copy()
	# set up detectors
	detector = CascadedDetector()

	# detection
	for i,r in enumerate(detector.detect(img)):
		x0 = r.left()
		y0 = r.top()
		x1 = r.right()
		y1 = r.bottom()
		
		cv2.rectangle(imgOut, (x0,y0),(x1,y1),(0,255,0),1)
		face = img[y0:y1,x0:x1]
		cv2.imshow('faces'+str(x0), face)

	# display image or write to file
	if outFileName is None:
		cv2.imshow('faces', imgOut)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else:
		cv2.imwrite(outFileName, imgOut) 
