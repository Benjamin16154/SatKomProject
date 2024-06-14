

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 21:01:16 2024

@author: Benjamin
"""

from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
#from mylib.mailer import Mailer
from mylib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest
#from pykalman import KalmanFilter

#python main.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4
#kf = KalmanFilter(initial_state_mean=[0, 0], n_dim_obs=2)
t0 = time.time()

def is_above_boundary_line(x, y, slope, intercept):
    return y < (slope * x + intercept)

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=False,
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", type=str,
        help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
        help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=1,
        help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=4,
        help="# of skip frames between detections")
    args = vars(ap.parse_args())

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    if not args.get("input", False):
        print("[INFO] Starting the live stream..")
        vs = VideoStream(config.url).start()
        time.sleep(2.0)
    else:
        print("[INFO] Starting the video..")
        vs = cv2.VideoCapture(args["input"])

    writer = None
    W = None
    H = None

    ct = CentroidTracker(maxDisappeared=300, maxDistance=200)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    totalDown = 0
    totalUp = 0
    x = []
    empty = []
    empty1 = []

    fps = FPS().start()

    if config.Thread:
        vs = thread.ThreadingClass(config.url)

    while True:
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame
        if frame is None:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]
            x1, y1 = 0, int(H * 0.9)
            x2, y2 = int(W * 0.85), 0
            boundary_line_slope = (y2 - y1) / (x2 - x1)
            boundary_line_intercept = y1 - boundary_line_slope * x1

        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

        status = "Waiting"
        rects = []

        if totalFrames % args["skip_frames"] == 0:
            status = "Detecting"
            trackers = []

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.2:
                    idx = int(detections[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    trackers.append(tracker)
        else:
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (0, int(H * 0.9)), (int(W * 0.85), 0), (255, 0, 0), 3)

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
               #y = [c[1] for c in to.centroids]
               #direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    if len(to.centroids) > 1:
                        prev_position = to.centroids[-2]
                        if is_above_boundary_line(prev_position[0], prev_position[1], boundary_line_slope, boundary_line_intercept) and not is_above_boundary_line(centroid[0], centroid[1], boundary_line_slope, boundary_line_intercept):
                            totalDown += 1
                            empty1.append(totalDown)
                            to.counted = True
                        elif not is_above_boundary_line(prev_position[0], prev_position[1], boundary_line_slope, boundary_line_intercept) and is_above_boundary_line(centroid[0], centroid[1], boundary_line_slope, boundary_line_intercept):
                            totalUp += 1
                            empty.append(totalUp)
                            to.counted = True

            trackableObjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        info = [
            ("Exit", totalUp),
            ("Enter", totalDown),
           # ("Status", status),
        ]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if writer is not None:
            writer.write(frame)
            
            
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        totalFrames += 1
        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    if writer is not None:
        writer.release()

    if not args.get("input", False):
        vs.stop()
    else:
        vs.release()

    cv2.destroyAllWindows()

    d = [datetime.datetime.now()]
    dts = [ts.strftime("%A %d %B %Y %I:%M:%S%p") for ts in d]
    export_data = zip_longest(*[dts, empty, empty1], fillvalue='')

    with open('Log.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(("End Time", "In", "Out"))
        writer.writerows(export_data)

run()
