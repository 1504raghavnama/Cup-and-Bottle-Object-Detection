import cv2
import numpy as np
import tensorflow as tf

# Step 1: Load TFLite model
print("[INFO] Loading model...")
try:
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
except Exception as e:
    print("[ERROR] Failed to load model:", e)
    exit()

# Step 2: Load labels
try:
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    print("[INFO] Labels loaded:", labels)
except Exception as e:
    print("[ERROR] Failed to load labels:", e)
    exit()

# Step 3: Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Step 4: Open webcam
print("[INFO] Accessing webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

print("[INFO] Webcam opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read from webcam.")
        break

    # Resize and preprocess image
    img = cv2.resize(frame, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img_rgb, axis=0).astype(np.float32) / 255.0

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get prediction
    prediction = np.squeeze(output_data)
    predicted_idx = np.argmax(prediction)
    confidence = prediction[predicted_idx]
    label = f"{labels[predicted_idx]}: {confidence:.2f}"

    # Display on frame
    cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Real-Time Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()