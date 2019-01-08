#!/usr/bin/env/python
# coding: utf-8

import cv2
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream
from matplotlib import pyplot as plt
from multiprocessing import Process
from multiprocessing import Queue
import imutils
import threading
import time

tracker = None

tracker = cv2.TrackerMOSSE_create()
is_initialized = False

def classify_frame(inputQueue, outputQueue,rl, cnd):
	while True:
		'''rl.acquire()
		while inputQueue.empty():
			cnd.wait()
		rl.release()'''
		if not inputQueue.empty():
			frame = inputQueue.get()
			# frame = cv2.resize(frame, (300, 300))
			face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
			faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
			outputQueue.put(faces)

inputQueue = Queue(1)
outputQueue = Queue(1)
faces = None

rl = threading.RLock()
cnd = threading.Condition(rl)
p = Process(target=classify_frame,args=(inputQueue,outputQueue,rl, cnd))
p.daemon = True
p.start()

vs = VideoStream(src=0).start()
time.sleep(1.0)
fps = None
initBB = None

insert_time = time.time() - 2.0

while True:
	frame = vs.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	if inputQueue.empty() and time.time() - insert_time > 2.0:
		#print("insert")
		'''rl.acquire()
		cnd.notify()
		rl.release()'''
		insert_time = time.time()
		inputQueue.put(gray)
	
	if not outputQueue.empty():
		faces = outputQueue.get() 
		
	if initBB is not None:
		(success, box) = tracker.update(gray)
		#print('tracker box: {', box, '}, success: {', success, '}')
		if success:
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

	if faces is not None:
		if len(faces) != 0: 
			#print(faces[0])
			initBB = tuple(faces[0])
			#if len(initBB) is 4:
			tracker = cv2.TrackerMOSSE_create()
			tracker.init(frame, initBB)
			#		is_initialized = True
			faces = None

	cv2.imshow('frame', frame)
	
	if fps is not None:
		H, W = frame.shape[:2]
		cv2.putText(frame,\
					"FPS {:.2f}".format(fps.fps()),\
					(10, H),\
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2\
					)
	
	key = cv2.waitKey(1) & 0xff
	if key == ord('q'):
		break
		
	if key == ord("s"):
		initBB = cv2.selectROI("Frame", gray, fromCenter=False, showCrosshair=True)
		fps = FPS().start()
		print(initBB)
		tracker.init(frame, initBB)

cv2.destroyAllWindows()
