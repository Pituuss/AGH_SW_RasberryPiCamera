#!/usr/bin/env/python
# coding: utf-8

import threading
import time
from multiprocessing import Queue
from threading import Thread

import cv2
from imutils.video import VideoStream

is_initialized = False
sem = threading.Event()
kill = True
inputQueue = Queue(1)
outputQueue = Queue(1)
faces = None
vs = VideoStream(src=0).start()
time.sleep(1.0)
initBB_1 = None
initBB_2 = None
feed_time = 1.0
insert_time = time.time() - feed_time
offset = 20
tracker = None
face_1 = None
face_2 = None


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

    if initBB_1 is not None:
        (success, box) = tracker_1.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (max(x - offset, 0), max(y - offset, 0)), (x + w + offset, y + h + offset),
                          (0, 255, 0), 2)
            x01, x11, y01, y11 = (max(y - offset, 0), y + h + offset, max(x - offset, 0), x + w + offset)
            face_1 = frame[x01:x11, y01:y11, :].copy()

    if initBB_2 is not None:
        (success, box) = tracker_2.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (max(x - offset, 0), max(y - offset, 0)), (x + w + offset, y + h + offset),
                          (0, 255, 0), 2)
            x02, x12, y02, y12 = (max(y - offset, 0), y + h + offset, max(x - offset, 0), x + w + offset)
            face_2 = frame[x02:x12, y02:y12, :].copy()

    if faces is not None:
        if len(faces) > 1:
            initBB_1 = tuple(faces[0])
            tracker_1 = cv2.TrackerKCF_create()
            tracker_1.init(frame, initBB_1)
            initBB_2 = tuple(faces[1])
            tracker_2 = cv2.TrackerKCF_create()
            tracker_2.init(frame, initBB_2)

    try:
        if face_1 is not None and face_2 is not None:
            frame[x01:x11, y01:y11, :] = cv2.resize(face_2, (x11 - x01, y11 - y01), .5, .5,
                                                    interpolation=cv2.INTER_CUBIC)
            frame[x02:x12, y02:y12, :] = cv2.resize(face_1, (x12 - x02, y12 - y02), .5, .5,
                                                    interpolation=cv2.INTER_CUBIC)

            face_1 = face_2 = None
    except:
        print('error occurred')

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

p.kill = False
sem.set()
p.join()

cv2.destroyAllWindows()
