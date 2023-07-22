import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model

model_path = "model2.h5"

def load_model_from_h5(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Inisialisasi model dari file .h5
model_path = "model2.h5"
model = load_model_from_h5(model_path)

IMG_SIZE = 150

def real_time_detection(model):
    # Start video capture from the camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess the frame for prediction
        frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
        frame_expanded = np.expand_dims(frame_resized, axis=0) / 255.0

        # Make the prediction
        prediction = model.predict(frame_expanded)
        class_idx = np.argmax(prediction[0])
        class_label = "organic" if class_idx == 0 else "inorganic"

        # Add class label to the frame
        cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with the prediction
        cv2.imshow("Deteksi Sampah Organik dan Anorganik", frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_detection(model)