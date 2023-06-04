# https://automaticaddison.com/how-to-detect-objects-in-video-using-mobilenet-ssd-in-opencv/
import cv2
import numpy as np

RESIZED_DIMENSIONS = (300, 300)  # Dimensions that SSD was trained on.
IMG_NORM_RATIO = 0.007843  # In grayscale a pixel can range between 0 and 255

# Load the pre-trained neural network
neural_network = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt',
                                          'MobileNetSSD_deploy.caffemodel')

# List of categories and classes
categories = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird',
              4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
              9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog',
              13: 'horse', 14: 'motorbike', 15: 'person',
              16: 'pottedplant', 17: 'sheep', 18: 'sofa',
              19: 'train', 20: 'tvmonitor'}

classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Create the bounding boxes
bbox_colors = np.random.uniform(255, 0, size=(len(categories), 3))

cap = cv2.VideoCapture(0)
# Process the video
while cap.isOpened():

    # Capture one frame at a time
    success, frame = cap.read()

    # Do we have a video frame? If true, proceed.
    if success:

        # Capture the frame's height and width
        (h, w) = frame.shape[:2]

        # Create a blob. A blob is a group of connected pixels in a binary
        # frame that share some common property (e.g. grayscale value)
        # Preprocess the frame to prepare it for deep learning classification
        frame_blob = cv2.dnn.blobFromImage(cv2.resize(frame, RESIZED_DIMENSIONS),
                                           IMG_NORM_RATIO, RESIZED_DIMENSIONS, 127.5)

        # Set the input for the neural network
        neural_network.setInput(frame_blob)

        # Predict the objects in the image
        neural_network_output = neural_network.forward()

        # Put the bounding boxes around the detected objects
        for i in np.arange(0, neural_network_output.shape[2]):

            confidence = neural_network_output[0, 0, i, 2]

            # Confidence must be at least 30%
            if confidence > 0.30:
                idx = int(neural_network_output[0, 0, i, 1])
                if classes[idx]== "person":
                    bounding_box = neural_network_output[0, 0, i, 3:7] * np.array(
                        [w, h, w, h])

                    (startX, startY, endX, endY) = bounding_box.astype("int")
                    print(startX, startY, endX, endY)

                    crop_img = frame[startY: endY, startX:endX]
                    if crop_img.shape[0] != 0 and crop_img.shape[1] != 0:
                        cv2.imshow('person', crop_img)

                    label = "{}: {:.2f}%".format(classes[idx], confidence * 100)

                    cv2.rectangle(frame, (startX, startY), (
                        endX, endY), bbox_colors[idx], 2)

                    y = startY - 15 if startY - 15 > 15 else startY + 15

                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, bbox_colors[idx], 2)

            # Display the resulting frame
            cv2.imshow('frame', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
