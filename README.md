# QR Code Detection Assignment in Autonomous Robotics Course


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
```