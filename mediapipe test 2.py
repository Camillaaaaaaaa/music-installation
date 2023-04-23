import cv2
import mediapipe as mp
import math
from loopstation import LoopStation
import time


class BodyDetection:

    def __init__(self):
        self.loop_station = LoopStation()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        self.height = 700

        # music score
        height_start = 100
        height_stop = self.height - 100
        self.score_spacing = (height_stop - height_start) / 6
        # y position of score line including start and end point(which do not have lines drawn on them)
        self.scores_y_pos = [height_start]

        for i in range(1, 6):
            h = int(height_start + i * self.score_spacing)
            self.scores_y_pos.append(h)
        self.scores_y_pos.append(height_stop)

        self.nodes_pos = {"D4": 5, "E4": 4.5, "F4": 4, "G4": 3.5, "A4": 3, "C5": 2}
        self.nodes_y_pos = {}
        for node in self.nodes_pos.keys():
            self.nodes_y_pos[node] = int(height_start + (self.nodes_pos[node] + 0.5) * self.score_spacing)

        # nodes that should be drawn for each beat
        self.nodes = [[["A4", 0]], [["D4", 0]], [["E4", 0], ["C5", 0]], [["G4", 0]], [["C5", 0]],
                      [["C5", 0], ["F4", 0]], [["A4", 0]], [["D4", 0]]]
        self.nodes_selected = [[], [], [], [], [], [], [], []]

        print(self.scores_y_pos)

    def webcam_analysis(self):
        cap = cv2.VideoCapture(0)
        count = 1
        last_time = time.time()

        with self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                success, image = cap.read()

                # print("height", image.shape[0])
                # print("width", image.shape[1])

                if time.time() >= last_time + self.loop_station.len_beat:
                    last_time = time.time()
                    count = self.loop_station.play_tone(count)

                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                image = cv2.resize(image, (int(self.height / image.shape[0] * image.shape[1]), self.height),
                                   interpolation=cv2.INTER_LINEAR)

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                # Draw landmark annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.mp_drawing.draw_landmarks(
                    image,
                    results.face_landmarks,
                    self.mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style())
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles
                    .get_default_pose_landmarks_style())

                image_with_score = self.draw_lines(image)
                image_with_score = self.draw_nodes(image_with_score)

                if results.pose_landmarks:
                    # get landmark coordinates of nose
                    image_height, image_width, _ = image.shape
                    x_coodinate = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.NOSE].x * image_width
                    y_coodinate = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.NOSE].y * image_height

                    # print(x_coodinate, " , ", y_coodinate)

                    # between_lines = self.check_y_pos_between_lines(y_coodinate)
                    # print(between_lines)

                    # select a node with nose
                    # return node, index of beat
                    in_circle = self.check_pos_in_node([x_coodinate, y_coodinate], image_width)
                    if in_circle:
                        if in_circle[0] not in self.nodes_selected[in_circle[1]]:
                            self.nodes_selected[in_circle[1]].append(in_circle[0])
                            self.loop_station.set_tones(self.nodes_selected)

                            # color node in
                            for node in self.nodes[in_circle[1]]:
                                if node[0] == in_circle[0]:
                                    node[1] = 1

                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Holistic', cv2.flip(image_with_score, 1))

                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()

    def draw_lines(self, img):
        for i in range(1, len(self.scores_y_pos) - 1):
            h = self.scores_y_pos[i]
            cv2.line(img, (0, h), (img.shape[1], h), (0, 0, 0), 3)
        return img

    def draw_nodes(self, img):
        section_width = img.shape[1] / 8
        for index, section in enumerate(self.nodes):
            for node in section:
                pos = (int(img.shape[1] - section_width * (index + 0.5)), self.nodes_y_pos[node[0]])
                if node[1] == 0:
                    cv2.circle(img, pos, int(self.score_spacing / 2), (51, 0, 51), 3)
                else:
                    cv2.circle(img, pos, int(self.score_spacing / 2), (204, 0, 204), -1)
        return img

    def check_pos_in_node(self, pos, img_width):
        section_width = img_width / 8
        for index, section in enumerate(self.nodes):
            for node in section:
                x_pos = int(img_width - section_width * (index + 0.5))
                y_pos = self.nodes_y_pos[node[0]]
                radius = int(self.score_spacing / 2)

                if math.sqrt((pos[0] - x_pos) ** 2 + (pos[1] - y_pos) ** 2) <= radius:
                    return node[0], index
        return False

    def check_y_pos_between_lines(self, y_pos):
        for i in range(len(self.scores_y_pos) - 1):
            if self.scores_y_pos[i] < y_pos < self.scores_y_pos[i + 1]:
                return i
        return False


detection = BodyDetection()
detection.webcam_analysis()
