import cv2
import numpy as np


class VideoAnalysis:
    def __init__(self, mp_holistic):
        self.mp_holistic = mp_holistic

        self.head_pos = []
        self.results = 0

        # variable for detection people with mobilenet ssd
        self.RESIZED_DIMENSIONS = (300, 300)  # Dimensions that SSD was trained on.
        self.IMG_NORM_RATIO = 0.007843  # In grayscale a pixel can range between 0 and 255

        # Load the pre-trained neural network
        self.neural_network = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt',
                                                       'MobileNetSSD_deploy.caffemodel')

        # List of categories and classes
        self.categories = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird',
                           4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
                           9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog',
                           13: 'horse', 14: 'motorbike', 15: 'person',
                           16: 'pottedplant', 17: 'sheep', 18: 'sofa',
                           19: 'train', 20: 'tvmonitor'}

        self.classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow",
                        "diningtable", "dog", "horse", "motorbike", "person",
                        "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        # Create the bounding boxes
        self.bbox_colors = np.random.uniform(255, 0, size=(len(self.categories), 3))

    def analyze(self, image, holistic):
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        return results

    def update_head_pos(self, image_height, image_width, offset_x, offset_y, results):
        x_coodinate = results.pose_landmarks.landmark[
                          self.mp_holistic.PoseLandmark.NOSE].x * image_width + offset_x
        y_coodinate = results.pose_landmarks.landmark[
                          self.mp_holistic.PoseLandmark.NOSE].y * image_height + offset_y

        head_pos = [x_coodinate, y_coodinate]
        return head_pos

    def human_detected(self):
        return self.results.pose_landmarks

    def find_users(self, frame):
        people = []

        # Capture the frame's height and width
        (h, w) = frame.shape[:2]

        # Create a blob. A blob is a group of connected pixels in a binary
        # frame that share some common property (e.g. grayscale value)
        # Preprocess the frame to prepare it for deep learning classification
        frame_blob = cv2.dnn.blobFromImage(cv2.resize(frame, self.RESIZED_DIMENSIONS),
                                           self.IMG_NORM_RATIO, self.RESIZED_DIMENSIONS, 127.5)

        # Set the input for the neural network
        self.neural_network.setInput(frame_blob)

        # Predict the objects in the image
        neural_network_output = self.neural_network.forward()

        # Put the bounding boxes around the detected objects
        for i in np.arange(0, neural_network_output.shape[2]):

            confidence = neural_network_output[0, 0, i, 2]

            # Confidence must be at least 30%
            if confidence > 0.60:
                idx = int(neural_network_output[0, 0, i, 1])
                if self.classes[idx] == "person":
                    bounding_box = neural_network_output[0, 0, i, 3:7] * np.array(
                        [w, h, w, h])

                    (startX, startY, endX, endY) = bounding_box.astype("int")

                    if startX < 0:
                        startX = 0

                    if startY < 0:
                        startY = 0

                    if endX < 0:
                        endX = 0

                    if endY < 0:
                        endY = 0

                    crop_img = frame[startY: endY, startX:endX]
                    """if crop_img.shape[0] != 0 and crop_img.shape[1] != 0:
                        cv2.imshow('person', crop_img)"""

                    people.append([crop_img, startX, startY])

        return people

    def check_head_in_note(self, current_instrument, notes_columns, notes_rows, head_pos):
        c = -1
        r = -1

        # find column
        for i in range(len(notes_columns[current_instrument]) - 1):
            if notes_columns[current_instrument][i] < head_pos[0] < notes_columns[current_instrument][i + 1]:
                c = i

        # find row
        for i in range(len(notes_rows) - 1):
            if notes_rows[i] < head_pos[1] < notes_rows[i + 1]:
                r = i / 2

        if c != -1 and r != -1:
            return [c, r]

        return False

    def get_face_size(self, image_height, image_width, results):
        x_ear_l = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_EAR].x * image_width
        y_ear_l = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_EAR].y * image_height
        x_ear_r = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_EAR].x * image_width
        y_ear_r = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_EAR].y * image_height

        radius_head = int(((x_ear_r - x_ear_l) ** 2 + (y_ear_r - y_ear_l) ** 2) ** 0.5 * 0.65)
        return radius_head

    def detect_shirt_color(self, image, results):
        # get part of image of upper body
        image_height, image_width, _ = image.shape
        left_shoulder = [
            int(results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width),
            int(results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height)]
        right_shoulder = [
            int(results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image_width),
            int(results.pose_landmarks.landmark[
                    self.mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image_height)]
        left_hip = [int(results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_HIP].x * image_width),
                    int(results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_HIP].y * image_height)]
        right_hip = [int(results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_HIP].x * image_width),
                     int(results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_HIP].y * image_height)]

        crop_width = int(left_shoulder[0] - right_shoulder[0])
        crop_height = int(left_hip[0] - right_shoulder[1])
        crop_img = image[right_shoulder[1]:right_shoulder[1] + crop_height,
                   right_shoulder[0]:right_shoulder[0] + crop_width]

        if crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
            # cv2.imshow("cropped", crop_img)

            # reduce resolution to make it faster
            try:
                crop_img = cv2.resize(crop_img, (int(crop_img.shape[1] / 3), int(crop_img.shape[0] / 3)),
                                      interpolation=cv2.INTER_LINEAR)
            except:
                print("could not crop image")

            # increase saturation
            # https://9to5answer.com/how-to-change-saturation-values-with-opencv
            """if self.increase_saturation:
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV).astype("float32")

                (h, s, v) = cv2.split(crop_img)
                s = s * 2
                s = np.clip(s, 0, 255)
                crop_img = cv2.merge([h, s, v])

                crop_img = cv2.cvtColor(crop_img.astype("uint8"), cv2.COLOR_HSV2BGR)"""

                # cv2.imshow("saturated", crop_img)
            # get dominant color of image
            # https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv

            pixels = np.float32(crop_img.reshape(-1, 3))
            n_colors = 5
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
            _, counts = np.unique(labels, return_counts=True)
            dominant = np.uint8(palette[np.argmax(counts)])
            # print("color", (dominant[0], dominant[1], dominant[2]))

            return (int(dominant[0]), int(dominant[1]), int(dominant[2]))
