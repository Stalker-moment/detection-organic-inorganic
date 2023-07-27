import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model

# Load the pre-trained waste classification model from the H5 file
model_path = "model2.h5"
model = load_model(model_path)

IMG_SIZE = 150

def classify_waste_object(roi):
    # Preprocess the ROI for waste classification
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_expanded = np.expand_dims(roi_resized, axis=0) / 255.0

    # Make the prediction for the waste object
    prediction = model.predict(roi_expanded)
    class_idx = np.argmax(prediction[0])
    class_label = "organic" if class_idx == 0 else "inorganic"

    return class_label

def real_time_tracking():
    # Start video capture from the camera
    cap = cv2.VideoCapture(0)

    # Load the Haar Cascade Classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Perform waste classification on the detected faces (if any)
        for (x, y, w, h) in faces:
            # Display "Manusia" label for face
            cv2.putText(frame, "Manusia", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Assuming the waste object is in the right half of the face_area
            roi = frame[y:y+h, x+w//2:x+w]

            # Check if the ROI is not empty
            if roi.size != 0:
                # Classify the waste object as "organic" or "inorganic"
                waste_label = classify_waste_object(roi)
                cv2.putText(frame, waste_label, (x+w//2, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame with waste classification (if any)
        cv2.imshow("Waste Classification", frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_tracking()
