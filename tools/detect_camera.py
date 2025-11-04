import cv2
import time

def show_camera(index, duration=100):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Camera {index} not available.")
        return False
    print(f"Showing camera {index} for {duration} seconds. Press any key to continue.")
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f"Camera {index}", frame)
        if cv2.waitKey(1) != -1 or (time.time() - start) > duration:
            break
    cap.release()
    cv2.destroyAllWindows()
    return True

def main():
    for idx in range(50):
         show_camera(idx)

if __name__ == "__main__":
    main()