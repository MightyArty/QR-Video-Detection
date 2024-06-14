# QR Code Detection Assignment in Autonomous Robotics Course

## Description
The provided code is a Aruco code detection algorithm, which for a given video/image file, detects the aruco codes that is located within the video/image, marks the aruco code with a green rectangle with the aruco ID and writes down the 2D information of each given [Frame, Aruco ID] to a csv file located under `Out` folder.

In addition, it prints to the terminal each Aruco code that it detects with it's ID.

## How to run
```bash
# Clone the repository
$ git clone https://github.com/MightyArty/QR-Video-Detection.git
# Enter the repository
$ cd QR-Video-Detection
# Install the libraries
$ pip install -r requirements.txt
# To run the QR-Image detection run the following
$ python detect_aruco_images.py --image Images/frame_1.png --type DICT_4X4_100
# Where "framge_1.png" can be changed to any other image containing a QR code.
# To run the QR-Video detection run the following
$ python detect_aruco_video.py --type DICT_5X5_100 --video Videos/challengeB.mp4
# After the run, you will see the output.csv file under the "Out" folder.
```