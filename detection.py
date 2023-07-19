import cv2
import RPi.GPIO as GPIO
import time

# Define GPIO pins for servomotors
SERVO_PIN_1 = 18
SERVO_PIN_2 = 19

# Load the pre-trained model for object detection
net = cv2.dnn.readNetFromTensorflow('path_to_frozen_inference_graph.pb', 'path_to_graph.pbtxt')

# List of labels for the detected objects
class_labels = ['organik', 'nonorganik']

# Initialize servomotor
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN_1, GPIO.OUT)
GPIO.setup(SERVO_PIN_2, GPIO.OUT)
servo_1 = GPIO.PWM(SERVO_PIN_1, 50)  # PWM frequency = 50Hz
servo_2 = GPIO.PWM(SERVO_PIN_2, 50)
servo_1.start(0)
servo_2.start(0)

def detect_trash(frame):
    # Perform object detection
    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detected objects
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        # Filter out weak detections and consider only organik and nonorganik labels
        if confidence > 0.5 and class_id in [1, 2]:
            # Get the bounding box coordinates
            box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            x, y, w, h = box.astype(int)

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            label = f'{class_labels[class_id - 1]}: {confidence:.2f}'
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Control servomotors based on detected class
            if class_id == 1:
                servo_1.ChangeDutyCycle(7.5)  # Rotate servo 1 (organik)
                time.sleep(1)
                servo_1.ChangeDutyCycle(0)
            elif class_id == 2:
                servo_2.ChangeDutyCycle(7.5)  # Rotate servo 2 (nonorganik)
                time.sleep(1)
                servo_2.ChangeDutyCycle(0)

    return frame

# Open the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    # Detect trash and control servomotors
    processed_frame = detect_trash(frame)

    # Display the frame
    cv2.imshow('Trash Detection', processed_frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture, stop the servomotors, and clean up GPIO
cap.release()
servo_1.stop()
servo_2.stop()
GPIO.cleanup()
cv2.destroyAllWindows()
