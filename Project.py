#!/usr/bin/env/python
# coding: utf-8

import cv2
from imutils.video import VideoStream
from multiprocessing import Queue
import threading
import time
from threading import Thread

is_initialized = False
sem = threading.Event()
kill = True
inputQueue = Queue(1)
outputQueue = Queue(1)
faces = None
vs = VideoStream(src=0).start()
time.sleep(1.0)
initBB = None
feed_time = 1.0
insert_time = time.time() - feed_time
edges = None
offset = 20
tracker = None


def classify_frame(input_queue, output_queue, sem, kill):
    time.sleep(2.0)
    t = threading.currentThread()
    while getattr(t, "kill", True):
        sem.wait()
        frame = input_queue.get()
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50),
                                              maxSize=(250, 250))
        output_queue.put(faces)
        sem.clear()


p = Thread(target=classify_frame, args=(inputQueue, outputQueue, sem, kill))
p.start()
while True:
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not sem.is_set() and time.time() - insert_time > feed_time:
        insert_time = time.time()
        inputQueue.put(gray)
        sem.set()

    if not outputQueue.empty():
        faces = outputQueue.get()

    if initBB is not None:
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            x0, x1, y0, y1 = (max(y - offset, 0), y + h + offset, max(x - offset, 0), x + w + offset)
            edges = cv2.Canny(frame[x0:x1, y0:y1], 50, 50)
            cv2.rectangle(frame, (max(x - offset, 0), max(y - offset, 0)), (x + w + offset, y + h + offset),
                          (0, 255, 0), 2)

    if faces is not None:
        if len(faces) != 0:
            initBB = tuple(faces[0])
            tracker = cv2.TrackerBoosting_create()
            tracker.init(frame, initBB)
            faces = None

    if edges is not None:
        frame[x0:x1, y0:y1, 2] = edges
        frame[x0:x1, y0:y1, 0] = edges
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

p.kill = False
sem.set()
p.join()

cv2.destroyAllWindows()
