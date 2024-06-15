"""
Detect ArUCo tags from a video file and save the output in a CSV file, in addition to displaying the detected tags along with the tag ID and corner points.

How to run:
python3 detect_from_video.py --type DICT_4X4_100 --video Videos/challengeB.mp4
"""

import csv
import os
import numpy as np
from utils import ARUCO_DICT, aruco_display, CAMERA_PARAMS
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
arucoParams = cv2.aruco.DetectorParameters_create()

output_folder = 'Out'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
csv_file = open('Out/video_output.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame ID', 'Marker ID', 'Top Left', 'Top Right', 'Bottom Right', 'Bottom Left', 'Distance', 'Yaw', 'Pitch', 'Roll'])

# Camera parameters
video_resolution = CAMERA_PARAMS["resolution"]
fov = CAMERA_PARAMS["fov"]

# Compute focal length from FoV and resolution
focal_length_x = video_resolution[0] / (2 * np.tan(np.radians(fov / 2)))
focal_length_y = video_resolution[1] / (2 * np.tan(np.radians(fov / 2)))

camera_matrix = np.array([[focal_length_x, 0, video_resolution[0] / 2],
                          [0, focal_length_y, video_resolution[1] / 2],
                          [0, 0, 1]])
dist_coeffs = np.zeros((5, 1))  # Assuming no lens distortion

frame_id = 0

while True:
    ret, frame = video.read()
    if not ret:
        print("[Info] End of video file reached.")
        break
    frame_id += 1
    h, w, _ = frame.shape

    width = 1000
    height = int(width * (h / w))
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    
    # Undistort the frame
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)
    
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    if ids is not None:
        print(f"[Info] Frame {frame_id}: Detected {len(ids)} markers")
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

            csv_writer.writerow([frame_id, ids[i][0], tuple(topLeft), tuple(topRight), tuple(bottomRight), tuple(bottomLeft), distance, yaw, pitch, roll])

    detected_markers = aruco_display(corners, ids, rejected, frame)

    cv2.imshow("Image", detected_markers)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

csv_file.close()
cv2.destroyAllWindows()
video.release()
