import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from PIL import Image

app = FastAPI(title="Emotion Recognition API")

# Load models at startup
@app.on_event("startup")
def load_models():
    global yolo_model, emotion_model, emotion_labels
    try:
        yolo_model = YOLO('yolov8n-face.pt')  # Path to YOLO face detection model
        emotion_model = load_model('final_emotion_model.h5')  # Path to emotion recognition model
        emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']  # Update based on your model
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e

def preprocess_face(face_img):
    """
    Preprocess the face image for emotion recognition.
    Convert to grayscale, resize, normalize, and reshape.
    """
    try:
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))
        face_normalized = face_resized / 255.0
        face_reshaped = np.expand_dims(face_normalized, axis=0)
        face_reshaped = np.expand_dims(face_reshaped, axis=-1)
        return face_reshaped
    except Exception as e:
        print(f"Error in preprocessing face: {e}")
        return None

def annotate_frame(frame, detections):
    """
    Draw bounding boxes and emotion labels on the frame.
    """
    for detection in detections:
        x1, y1, x2, y2, cls_name, conf, emotion_label, emotion_conf = detection
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Prepare emotion text
        emotion_text = f"{emotion_label} {emotion_conf:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate text position
        text_x = int(x1)
        text_y = int(y1) - 10  # 10 pixels above the bounding box

        # Ensure text does not go above the frame
        if text_y - text_height - baseline < 0:
            text_y = int(y2) + text_height + baseline + 10

        # Draw rectangle background for text
        cv2.rectangle(frame, (text_x, text_y - text_height - baseline),
                      (text_x + text_width, text_y), (0, 255, 0), -1)

        # Put emotion text above bounding box
        cv2.putText(frame, emotion_text, (text_x, text_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    """
    Endpoint to process an image frame.
    Receives an image file, detects faces, recognizes emotions, annotates the image, and returns it.
    """
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Detect faces using YOLO
        results = yolo_model.predict(frame, conf=0.4)

        detections = []
        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                conf = box.conf[0].item() if box.conf is not None else 0.0
                cls_name = yolo_model.names[cls_id] if cls_id in yolo_model.names else "N/A"

                # Crop face from frame
                face_img = frame[int(y1):int(y2), int(x1):int(x2)]
                if face_img.size == 0:
                    continue  # Skip if face crop is empty

                # Preprocess and predict emotion
                preprocessed_face = preprocess_face(face_img)
                if preprocessed_face is None:
                    continue

                emotion_prediction = emotion_model.predict(preprocessed_face)
                emotion_label = emotion_labels[np.argmax(emotion_prediction)]
                emotion_conf = np.max(emotion_prediction)

                # Append detection info
                detections.append((x1, y1, x2, y2, cls_name, conf, emotion_label, emotion_conf))

        # Annotate frame with detections
        annotated_frame = annotate_frame(frame, detections)

        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        jpg_bytes = buffer.tobytes()

        return StreamingResponse(io.BytesIO(jpg_bytes), media_type="image/jpeg")

    except Exception as e:
        print(f"Error processing frame: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/", response_class=HTMLResponse)
def index():
    """
    Simple HTML page to upload an image and display the processed image.
    """
    return """
    <html>
        <head>
            <title>Emotion Recognition API</title>
        </head>
        <body>
            <h1>Upload an Image for Emotion Recognition</h1>
            <form action="/process_frame" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
