from ultralytics import YOLO
import cv2 as cv
import threading
import queue
import time
import numpy as np
from tensorflow.keras.models import load_model


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
                print("Không thể đọc khung hình từ camera")
                break

            # Mỗi 5 khung hình, gửi khung hình để phát hiện
            if self.frame_count % 5 == 0:
                if self.frame_queue.empty():
                    self.frame_queue.put(frame.copy())

            # Kiểm tra kết quả mới từ DetectionThread
            if not self.result_queue.empty():
                self.current_boxes = self.result_queue.get()

            # Vẽ bounding boxes và văn bản
            if self.current_boxes:
                for box in self.current_boxes:
                    x1, y1, x2, y2, cls_name, conf, emotion_label, emotion_conf = box
                    # Vẽ bounding box
                    cv.rectangle(frame, (int(x1), int(y1)),
                                 (int(x2), int(y2)), (255, 0, 0), 2)

                    # Chuẩn bị văn bản cho cảm xúc
                    emotion_text = f"{emotion_label} {emotion_conf:.2f}"
                    (e_text_width, e_text_height), e_baseline = cv.getTextSize(emotion_text, cv.FONT_HERSHEY_SIMPLEX,
                                                                               0.5, 1)

                    # Tính vị trí văn bản (trên bounding box)
                    text_x = int(x1)
                    text_y = int(y1) - 10  # Lùi lên 10 pixel từ đỉnh bounding box

                    # Đảm bảo văn bản không bị cắt khi ở đầu khung hình
                    if text_y - e_text_height - e_baseline < 0:
                        text_y = int(y2) + e_text_height + e_baseline + 10

                    # Vẽ nền cho văn bản cảm xúc (tuỳ chọn)
                    cv.rectangle(frame, (text_x, text_y - e_text_height - e_baseline),
                                 (text_x + e_text_width, text_y), (0, 255, 0), -1)

                    # Vẽ văn bản cảm xúc màu xanh lá cây
                    cv.putText(frame, emotion_text, (text_x, text_y - e_baseline),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

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
        self.model = YOLO('yolov8n-face.pt')  # Mô hình YOLO để phát hiện khuôn mặt
        self.emotion_model = load_model('final_emotion_model.h5')  # Tải mô hình nhận diện cảm xúc
        self.emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']  # Nhãn cảm xúc
        self.running = True

    def preprocess_face(self, face_img):
        # Chuyển đổi ảnh sang grayscale và resize về kích thước mong muốn (ví dụ: 48x48)
        face_gray = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
        face_resized = cv.resize(face_gray, (48, 48))
        face_normalized = face_resized / 255.0
        face_reshaped = np.expand_dims(face_normalized, axis=0)
        face_reshaped = np.expand_dims(face_reshaped, axis=-1)
        return face_reshaped

    def run(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                results = self.model.predict(frame, conf=0.4)

                current_boxes = []
                for result in results:
                    for box in result.boxes:
                        # Lấy tọa độ bounding box
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        # Lấy lớp và độ tin cậy
                        cls_id = int(box.cls[0]) if box.cls is not None else -1
                        conf = box.conf[0].item() if box.conf is not None else 0.0
                        # Lấy tên lớp từ mô hình (nếu mô hình hỗ trợ)
                        cls_name = self.model.names[cls_id] if cls_id in self.model.names else "N/A"

                        # Cắt khuôn mặt từ khung hình
                        face_img = frame[int(y1):int(y2), int(x1):int(x2)]
                        if face_img.size == 0:
                            continue  # Bỏ qua nếu không cắt được khuôn mặt

                        # Tiền xử lý và nhận diện cảm xúc
                        preprocessed_face = self.preprocess_face(face_img)
                        emotion_prediction = self.emotion_model.predict(preprocessed_face)
                        emotion_label = self.emotion_labels[np.argmax(emotion_prediction)]
                        emotion_conf = np.max(emotion_prediction)

                        # Tạo một tuple chứa tất cả thông tin
                        current_boxes.append((x1, y1, x2, y2, cls_name, conf, emotion_label, emotion_conf))

                # Cập nhật result_queue
                while not self.result_queue.empty():
                    self.result_queue.get()
                self.result_queue.put(current_boxes)

            time.sleep(0.01)

    def stop(self):
        self.running = False


def main():
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)

    # Tạo và khởi động các luồng
    camera_thread = CameraThread(frame_queue, result_queue)
    detection_thread = DetectionThread(frame_queue, result_queue)

    camera_thread.start()
    detection_thread.start()

    try:
        # Chờ cho đến khi luồng camera kết thúc (khi người dùng nhấn 'q')
        camera_thread.join()
    except KeyboardInterrupt:
        print("Đã bị người dùng ngắt")
    finally:
        # Dừng các luồng
        camera_thread.stop()
        detection_thread.stop()
        detection_thread.join()


if __name__ == "__main__":
    main()
