"""
Detect ArUCo tags from an image and display the detected tags along with the tag ID and corner points.

How to run:
python3 detect_from_image.py --image Images/frame_1.jpg --type DICT_4X4_100
"""

import csv
import os
import numpy as np
from utils import ARUCO_DICT, aruco_display, CAMERA_PARAMS
import argparse
import cv2
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image containing ArUCo tag")
ap.add_argument("-t", "--type", type=str, default="DICT_4X4_100", help="type of ArUCo tag to detect")
args = vars(ap.parse_args())

if args["image"] is None:
    print("[Error] Image file location is not provided")
    sys.exit(1)

print(f"Loading image from path: {args['image']}")
image = cv2.imread(args["image"])

if image is None:
    print(f"[Error] Could not open or find the image: {args['image']}")
    sys.exit(1)

h, w, _ = image.shape
width = 600
height = int(width * (h / w))
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

# verify that the supplied ArUCo tag exists and is supported by OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
    print(f"ArUCo tag type '{args['type']}' is not supported")
    sys.exit(0)

# load the ArUCo dictionary, grab the ArUCo parameters, and detect the markers
print("Detecting '{}' tags....".format(args["type"]))
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()
corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

output_folder = 'Out'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
frame_name = os.path.splitext(os.path.basename(args["image"]))[0]
csv_file_name = f'{frame_name}_image_output.csv'
csv_file = open(f'Out/{csv_file_name}', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Marker ID', 'Top Left', 'Top Right', 'Bottom Right', 'Bottom Left', 'Distance', 'Yaw', 'Pitch', 'Roll'])

# Camera parameters
image_resolution = CAMERA_PARAMS["resolution"]
fov = CAMERA_PARAMS["fov"]

# Compute focal length from FoV and resolution
focal_length_x = image_resolution[0] / (2 * np.tan(np.radians(fov / 2)))
focal_length_y = image_resolution[1] / (2 * np.tan(np.radians(fov / 2)))

camera_matrix = np.array([[focal_length_x, 0, image_resolution[0] / 2],
                          [0, focal_length_y, image_resolution[1] / 2],
                          [0, 0, 1]])
dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion

if ids is not None:
    for i in range(len(ids)):
        marker_corners = corners[i].reshape((4, 2))
        topLeft, topRight, bottomRight, bottomLeft = marker_corners

        # Estimate the pose of each marker
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corners.reshape(1, 4, 2), 0.05, camera_matrix, dist_coeffs)
        rvec, tvec = rvec[0,0,:], tvec[0,0,:]

        # Calculate yaw, pitch, roll
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            yaw = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            yaw = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0

        # Convert to degrees
        yaw = np.degrees(yaw)
        pitch = np.degrees(pitch)
        roll = np.degrees(roll)
        distance = np.linalg.norm(tvec)

        csv_writer.writerow([ids[i][0], tuple(topLeft), tuple(topRight), tuple(bottomRight), tuple(bottomLeft), distance, yaw, pitch, roll])

detected_markers = aruco_display(corners, ids, rejected, image)
cv2.imshow("Image", detected_markers)
cv2.waitKey(0)

csv_file.close()
cv2.destroyAllWindows()