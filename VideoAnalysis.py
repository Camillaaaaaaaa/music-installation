import cv2
import numpy as np


class VideoAnalysis:
    def __init__(self, mp_holistic):
        self.mp_holistic = mp_holistic

        self.head_pos = []
        self.results = 0

    def analyze(self, image, holistic):
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = holistic.process(image)

    def update_head_pos(self, image_height, image_width):
        x_coodinate = self.results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.NOSE].x * image_width
        y_coodinate = self.results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.NOSE].y * image_height

        self.head_pos = [x_coodinate, y_coodinate]

    def human_detected(self):
        return self.results.pose_landmarks

    def check_head_in_note(self, current_instrument, notes_columns, notes_rows):
        c = -1
        r = -1

        # find column
        for i in range(len(notes_columns[current_instrument])-1):
            if notes_columns[current_instrument][i] < self.head_pos[0] < notes_columns[current_instrument][i + 1]:
                c = i

        # find row
        for i in range(len(notes_rows)-1):
            if notes_rows[i] < self.head_pos[1] < notes_rows[i + 1]:
                r = i/2

        if c != -1 and r != -1:
            return [c, r]

        return False

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
            if self.increase_saturation:
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV).astype("float32")

                (h, s, v) = cv2.split(crop_img)
                s = s * 2
                s = np.clip(s, 0, 255)
                crop_img = cv2.merge([h, s, v])

                crop_img = cv2.cvtColor(crop_img.astype("uint8"), cv2.COLOR_HSV2BGR)

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
