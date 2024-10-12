from ultralytics import YOLO
import cv2 as cv
import threading
import queue
import time


class CameraThread(threading.Thread):
    def __init__(self, frame_queue, result_queue):
        threading.Thread.__init__(self)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.capture = cv.VideoCapture(0)
        self.running = True
        self.frame_count = 0
        self.current_boxes = []

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                print("Cannot read frame from camera")
                break

            # Every 5 frames, send frame for detection
            if self.frame_count % 5 == 0:
                if self.frame_queue.empty():
                    self.frame_queue.put(frame)

            # Check for new detection results
            if not self.result_queue.empty():
                self.current_boxes = self.result_queue.get()

            # Draw boxes
            if self.current_boxes:
                for box in self.current_boxes:
                    x1, y1, x2, y2 = box
                    cv.rectangle(frame, (int(x1), int(y1)),
                                 (int(x2), int(y2)), (255, 0, 0), 2)

            cv.imshow('Video', frame)
            self.frame_count += 1

            if cv.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

    def stop(self):
        self.running = False
        self.capture.release()
        cv.destroyAllWindows()


class DetectionThread(threading.Thread):
    def __init__(self, frame_queue, result_queue):
        threading.Thread.__init__(self)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.model = YOLO('yolov8n-face.pt')
        self.running = True

    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                results = self.model.predict(frame, conf=0.4)

                current_boxes = []
                for info in results:
                    for box in info.boxes:
                        current_boxes.append(box.xyxy[0].tolist())

                # Update result queue
                while not self.result_queue.empty():
                    self.result_queue.get()
                self.result_queue.put(current_boxes)

            time.sleep(0.01)

    def stop(self):
        self.running = False


def main():
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)

    # Create and start threads
    camera_thread = CameraThread(frame_queue, result_queue)
    detection_thread = DetectionThread(frame_queue, result_queue)

    camera_thread.start()
    detection_thread.start()

    # Wait for camera thread to finish (when user presses 'q')
    camera_thread.join()

    # Stop detection thread
    detection_thread.stop()


if __name__ == "__main__":
    main()