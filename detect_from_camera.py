"""
Assignment 2-B: Detecting an Aruco marker, and then giving directions to move the camera to align it at the same point from where the shot was taken.
Instructions:
1. Run the file
2. Place an Aruco marker at some point
3. Press 'c' to capture the data of the marker
4. Move the camera to another point
5. The program will give you directions to move the camera to align the current marker with the reference marker
"""

import numpy as np
from utils import ARUCO_DICT, aruco_display, CAMERA_PARAMS
import argparse
import cv2
import sys
import time

def give_direction(current_data, reference_data):
    """
    Give directions to move the camera to align the current marker with the reference marker.

    Parameters:
    ----------
    - `current_data`: data of the current marker
    - `reference_data`: data of the reference marker
    """
    direction = []
    if current_data['Distance'] > reference_data['Distance']:
        direction.append('Move closer')
    else:
        direction.append('Move farther')
    
    yaw_diff = current_data['Yaw'] - reference_data['Yaw']
    if yaw_diff > 5:
        direction.append('Turn left')
    elif yaw_diff < -5:
        direction.append('Turn right')
    
    pitch_diff = current_data['Pitch'] - reference_data['Pitch']
    if pitch_diff > 5:
        direction.append('Tilt down')
    elif pitch_diff < -5:
        direction.append('Tilt up')
    
    roll_diff = current_data['Roll'] - reference_data['Roll']
    if roll_diff > 5:
        direction.append('Rotate counterclockwise')
    elif roll_diff < -5:
        direction.append('Rotate clockwise')

    return ', '.join(direction) if direction else 'Stay'

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str, default="DICT_4X4_100", help="Type of ArUCo tag to detect")
args = vars(ap.parse_args())

if ARUCO_DICT.get(args["type"], None) is None:
    print(f"ArUCo tag type '{args['type']}' is not supported")
    sys.exit(0)

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    print("[Error] Could not access the camera.")
    sys.exit(1)

h, w = frame.shape[:2]
video_resolution = (w, h)

fov = 60

# Compute focal length from FoV and resolution
focal_length_x = video_resolution[0] / (2 * np.tan(np.radians(fov / 2)))
focal_length_y = video_resolution[1] / (2 * np.tan(np.radians(fov / 2)))

camera_matrix = np.array([[focal_length_x, 0, video_resolution[0] / 2],
                          [0, focal_length_y, video_resolution[1] / 2],
                          [0, 0, 1]])
dist_coeffs = np.zeros((5, 1))

captured_data = None
last_print_time = 0
print_interval = 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("[Error] Can't receive frame from camera. Exiting ...")
        break

    frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)

    corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    if ids is not None:
        for i in range(len(ids)):
            marker_corners = corners[i].reshape((4, 2))
            topLeft, topRight, bottomRight, bottomLeft = marker_corners

            marker_points = np.array([[-0.025, 0.025, 0], [0.025, 0.025, 0], [0.025, -0.025, 0], [-0.025, -0.025, 0]],
                                     dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(marker_points, marker_corners, camera_matrix, dist_coeffs)

            if success:
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

                yaw = np.degrees(yaw)
                pitch = np.degrees(pitch)
                roll = np.degrees(roll)
                distance = np.linalg.norm(tvec)

                current_data = {
                    'Marker ID': ids[i][0],
                    'Top Left': tuple(topLeft),
                    'Top Right': tuple(topRight),
                    'Bottom Right': tuple(bottomRight),
                    'Bottom Left': tuple(bottomLeft),
                    'Distance': distance,
                    'Yaw': yaw,
                    'Pitch': pitch,
                    'Roll': roll
                }

                if captured_data:
                    current_time = time.time()
                    if current_time - last_print_time >= print_interval:
                        direction = give_direction(current_data, captured_data)
                        print(f"[Direction] Marker {ids[i][0]}: {direction}")
                        last_print_time = current_time

    detected_markers = aruco_display(corners, ids, rejected, frame)

    cv2.imshow("Camera Feed", detected_markers)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        if ids is not None:
            captured_data = current_data
            print(f"[Capture] Captured data for Marker {captured_data['Marker ID']}")
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
