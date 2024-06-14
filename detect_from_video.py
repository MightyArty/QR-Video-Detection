"""
Detect ArUCo tags from a video file and save the output in a CSV file, in addition to displaying the detected tags along with the tag ID and corner points.
"""

import csv
import os
from utils import ARUCO_DICT, aruco_display
import argparse
import cv2
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path to the video file")
ap.add_argument("-t", "--type", type=str, default="DICT_4X4_100", help="Type of ArUCo tag to detect")
args = vars(ap.parse_args())

if args["video"] is None:
    print("[Error] Video file location is not provided")
    sys.exit(1)

video = cv2.VideoCapture(args["video"])

if ARUCO_DICT.get(args["type"], None) is None:
    print(f"ArUCo tag type '{args['type']}' is not supported")
    sys.exit(0)

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()

output_folder = 'Out'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
csv_file = open('Out/video_output.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame ID', 'Marker ID', 'Top Left', 'Top Right', 'Bottom Right', 'Bottom Left'])

frame_id = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    frame_id += 1
    h, w, _ = frame.shape

    width = 1000
    height = int(width * (h / w))
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        for i in range(len(ids)):
            marker_corners = corners[i].reshape((4, 2))
            topLeft, topRight, bottomRight, bottomLeft = marker_corners
            csv_writer.writerow([frame_id, ids[i][0], tuple(topLeft), tuple(topRight), tuple(bottomRight), tuple(bottomLeft)])

    detected_markers = aruco_display(corners, ids, rejected, frame)

    cv2.imshow("Image", detected_markers)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

csv_file.close()
cv2.destroyAllWindows()
video.release()