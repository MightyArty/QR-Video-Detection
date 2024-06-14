"""
Detect ArUCo tags from an image and display the detected tags along with the tag ID and corner points.
"""

import csv
import os
from utils import ARUCO_DICT, aruco_display
import argparse
import cv2
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image containing ArUCo tag")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to detect")
args = vars(ap.parse_args())

if args["image"] is None:
    print("[Error] Image file location is not provided")
    sys.exit(1)

image = cv2.imread(args["image"])
h, w, _ = image.shape
width = 600
height = int(width * (h / w))
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

if ARUCO_DICT.get(args["type"], None) is None:
    print(f"ArUCo tag type '{args['type']}' is not supported")
    sys.exit(0)

print("Detecting '{}' tags....".format(args["type"]))
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(dictionary=arucoDict, detectorParams=arucoParams)
corners, ids, rejected = detector.detectMarkers(image)

output_folder = 'Out'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
frame_name = os.path.splitext(os.path.basename(args["image"]))[0]
csv_file_name = f'{frame_name}_image_output.csv'
csv_file = open(f'Out/{csv_file_name}', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Marker ID', 'Top Left', 'Top Right', 'Bottom Right', 'Bottom Left'])

if ids is not None:
    for i in range(len(ids)):
        marker_corners = corners[i].reshape((4, 2))
        topLeft, topRight, bottomRight, bottomLeft = marker_corners
        csv_writer.writerow([ids[i][0], tuple(topLeft), tuple(topRight), tuple(bottomRight), tuple(bottomLeft)])

detected_markers = aruco_display(corners, ids, rejected, image)
cv2.imshow("Image", detected_markers)
cv2.waitKey(0)