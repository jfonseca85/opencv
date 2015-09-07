#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   This face detector is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
#
# COMPILING THE DLIB PYTHON INTERFACE
#   Dlib comes with a compiled python interface for python 2.7 on MS Windows. If
#   you are using another python version or operating system then you need to
#   compile the dlib python interface before you can use this file.  To do this,
#   run compile_dlib_python_module.bat.  This should work on any operating
#   system so long as you have CMake and boost-python installed.
#   On Ubuntu, this can be done easily by running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install -U scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import glob
import os
import sys

import cv2
import dlib
import numpy as np
from skimage import io


if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

cap = cv2.VideoCapture(0)

#for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
#    print("Processing file: {}".format(f))

# Create the correlation tracker - the object needs to be initialized
# before it can be used
tracker = dlib.correlation_tracker()
trackerShape = dlib.correlation_tracker()

#k = 0
while(1):
    
    # Take each frame
    _, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
#    img = io.imread(frame)

#   cv2.imshow('frame',frame)
#   cv2.setWindowTitle('Reconecimento facial')
    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(tracker.get_position())
#    win.add_overlay(trackerShape.get_position())
    

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(gray, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        
        # We need to initialize the tracker on the first frame
#        if k == 0:
            # Start a track on the juice box. If you look at the first frame you
            # will see that the juice box is contained within the bounding
            # box (74, 67, 112, 153).      
        tracker.start_track(img, dlib.rectangle(d.left(),d.top(),  d.right(),  d.bottom()))
        #just attempt to track from the previous frame
        tracker.update(img)

        # Draw the face landmarks on the screen.
        win.add_overlay(shape)
        win.add_overlay(d)


    #dlib.hit_enter_to_continue()
#    k = cv2.waitKey(5) & 0xFF
#    if k == 27:
#       break

cv2.destroyAllWindows()
