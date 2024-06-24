import cv2

def test_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[Error] Could not open camera {index}")
        return
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Camera Test", frame)
        cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

# Test default camera (index 0)
test_camera(0)

# Test second camera (index 1)
test_camera(1)
