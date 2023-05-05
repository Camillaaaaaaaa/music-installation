import cv2
import mediapipe as mp
import math
from loopstation import LoopStation
import pygame
import time
from note import Note
import numpy as np
import keyboard


class BodyDetection:

    def __init__(self):
        # different versions
        self.amount_beats = 8
        self.bpm = 140
        self.score_width = 1000
        self.height = 900
        self.increase_saturation = False
        self.detect_shirt_c = True
        self.show_other_tracks = True
        self.outline = True
        self.bigger_selection_space = True
        self.loop_instruments = False
        self.base_drum_track = False

        self.loop_duration = 15
        self.last_loop = 0

        self.amount_instruments = 8
        self.instruments = ["Drums", "Piano", "Bass", "Guitar", "xylophone", "Piano Chords", "Guitar Chords"]

        self.current_instrument = 0

        self.loop_station = LoopStation(self.amount_beats, self.bpm)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic

        self.note_radius = 0
        self.sec_until_selected = 1
        self.time_until_selecting = 0.4
        self.time_selection = 0
        self.last_note = [0, 0]

        # music score
        height_start = 70
        height_stop = self.height - 250
        self.score_xpos = 0
        self.section_width = 0
        self.score_spacing = (height_stop - height_start) / 6
        # y position of score line including start and end point(which do not have lines drawn on them)
        self.scores_y_pos = [height_start]

        for i in range(1, 6):
            h = int(height_start + i * self.score_spacing)
            self.scores_y_pos.append(h)
        self.scores_y_pos.append(height_stop)

        possible_notes_pos = [5, 4, 3, 2, 1, 0]

        # "D4": 5, "E4": 4.5, "F4": 4, "G4": 3.5, "A4": 3, "C5": 2
        self.notes = []
        for a in range(self.amount_instruments):
            self.notes.append([])
            for i in range(self.loop_station.amount_beats):
                for pos in possible_notes_pos:
                    self.notes[a].append(Note(pos, i))

        for i in self.notes:
            for note in i:
                note.y_pos = int(height_start + (note.score_pos + 0.5) * self.score_spacing)

        self.notes_selected = []
        for a in range(self.amount_instruments):
            self.notes_selected.append([])
            for i in range(self.loop_station.amount_beats):
                self.notes_selected[a].append([])

        self.loop_station.set_tones(self.notes_selected)

        self.frame_counter = 0

        # colors
        self.head_color = (159, 66, 94)
        self.head_outline_color = (0, 0, 0)

        self.window_name = "loop station"

        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.bpm_keys = [0, 0]
        self.drum_key = 0
        self.played_half_beat = False

        # music loops
        self.current_pos_loops = 0
        self.music_loops_1 = [5, 6, 4]
        self.music_loops_2 = [0, 6, 1]
        self.current_loops = self.music_loops_1

    def webcam_analysis(self):
        print("before video capture")
        cap = cv2.VideoCapture(0)
        print("video capture")
        count = 1
        last_time = time.time()
        self.last_loop = time.time()

        with self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                success, image = cap.read()

                # print("height", image.shape[0])
                # print("width", image.shape[1])

                # loop through instruments
                if self.loop_instruments:
                    if time.time() >= self.last_loop + self.loop_duration:
                        self.last_loop = time.time()
                        self.current_pos_loops += 1
                        if self.current_pos_loops > len(self.current_loops) - 1:
                            self.current_pos_loops = 0
                        self.current_instrument = self.current_loops[self.current_pos_loops]

                        # delete last track of instrument
                        self.notes_selected[self.current_instrument] = []
                        for _ in range(self.loop_station.amount_beats):
                            self.notes_selected[self.current_instrument].append([])
                        self.loop_station.set_tones(self.notes_selected)

                        # reset notes of instrument
                        for note in self.notes[self.current_instrument]:
                            note.status = 0

                # play note on beat
                if time.time() >= last_time + self.loop_station.len_beat:
                    last_time = time.time()
                    count = self.loop_station.play_tone(count)
                    self.played_half_beat = False
                    print(count)
                if time.time() >= last_time + self.loop_station.len_beat / 2 and not self.played_half_beat:
                    self.played_half_beat = True
                    self.loop_station.play_half_beat(count)

                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                image = cv2.resize(image, (int(self.height / image.shape[0] * image.shape[1]), self.height),
                                   interpolation=cv2.INTER_LINEAR)

                # image = self.undistort(image)

                # set dimensions of score
                if self.score_xpos == 0:
                    self.score_xpos = int((image.shape[1] - self.score_width) / 2)
                    self.section_width = self.score_width / self.loop_station.amount_beats
                    print(self.score_xpos)
                    print(image.shape[1])

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                # Draw landmark annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    image = self.draw_user(image, results)
                    if self.detect_shirt_c:
                        if self.frame_counter > 20:
                            self.detect_shirt_color(image, results)
                            self.frame_counter = 0
                        else:
                            self.frame_counter += 1

                    # get landmark coordinates of nose
                    image_height, image_width, _ = image.shape
                    x_coodinate = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.NOSE].x * image_width
                    y_coodinate = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.NOSE].y * image_height

                    # select a note with nose
                    # return file name, index of beat
                    in_circle = self.check_pos_in_note([x_coodinate, y_coodinate])
                    if in_circle:
                        for note in self.notes[self.current_instrument]:
                            if in_circle[0] == note.y_pos and in_circle[1] == note.beat:
                                if note.status == 0:
                                    note.status = 1
                                    self.time_selection = time.time()
                                if self.last_note[0] == in_circle[0] and self.last_note[1] == in_circle[1]:
                                    time_passed = time.time() - self.time_selection
                                    if note.status == 1 and time_passed > self.time_until_selecting:
                                        note.status = 2
                                        self.note_radius = 0
                                        self.time_selection = time.time()
                                    if note.status == 2:
                                        if time_passed > self.sec_until_selected:
                                            note.status = 3
                                            note.status_3_color = self.head_color
                                            self.notes_selected[self.current_instrument][note.beat].append(
                                                note.score_pos)
                                        else:
                                            self.note_radius = (
                                                                       time.time() - self.time_selection) / self.sec_until_selected * 360

                                self.loop_station.set_tones(self.notes_selected)
                                self.last_note = in_circle

                image_with_score = self.draw_lines(image)
                image_with_score = self.draw_notes(image_with_score)

                # draw moving time line
                line_x_pos = int(self.score_xpos + self.score_width - (
                        (time.time() - last_time) / self.loop_station.len_beat + count - 1) * self.section_width)
                cv2.line(image_with_score, (line_x_pos, 0), (line_x_pos, image_with_score.shape[1]),
                         (0, 0, 0), 2)

                image_with_score = cv2.flip(image_with_score, 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image_with_score, self.instruments[self.current_instrument], (10, 70), font, 2.5,
                            (0, 0, 0), 10, cv2.LINE_AA)
                cv2.putText(image_with_score, self.instruments[self.current_instrument], (10, 70), font, 2.5,
                            (255, 255, 255), 3, cv2.LINE_AA)

                cv2.putText(image_with_score, "BPM: " + str(self.bpm), (image_with_score.shape[1] - 450, 70), font, 2.5,
                            (0, 0, 0), 10, cv2.LINE_AA)
                cv2.putText(image_with_score, "BPM: " + str(self.bpm), (image_with_score.shape[1] - 450, 70), font, 2.5,
                            (255, 255, 255), 3, cv2.LINE_AA)

                cv2.imshow(self.window_name, image_with_score)

                for i in range(self.amount_instruments + 1):
                    if keyboard.is_pressed(i + 1):
                        print(i)
                        self.current_instrument = i - 1
                        print("current", self.current_instrument)

                        # delete last track of instrument
                        self.notes_selected[self.current_instrument] = []
                        for _ in range(self.loop_station.amount_beats):
                            self.notes_selected[self.current_instrument].append([])
                        self.loop_station.set_tones(self.notes_selected)

                        # reset notes of instrument
                        for note in self.notes[self.current_instrument]:
                            note.status = 0

                if keyboard.is_pressed("0"):
                    self.loop_instruments = False

                if keyboard.is_pressed("8"):
                    self.loop_instruments = True
                    self.current_loops = self.music_loops_1
                    self.current_pos_loops = 0
                    self.current_instrument = self.current_loops[self.current_pos_loops]

                if keyboard.is_pressed("9"):
                    self.loop_instruments = True
                    self.current_loops = self.music_loops_2
                    self.current_pos_loops = 0
                    self.current_instrument = self.current_loops[self.current_pos_loops]

                # change bpm with q and w keys
                if keyboard.is_pressed("q"):
                    self.bpm_keys[0] = 1

                if not keyboard.is_pressed("q") and self.bpm_keys[0] == 1:
                    self.bpm -= 5
                    self.bpm_keys[0] = 0
                    self.loop_station.set_bpm(self.bpm)

                if keyboard.is_pressed("w"):
                    self.bpm_keys[1] = 1

                if not keyboard.is_pressed("w") and self.bpm_keys[1] == 1:
                    self.bpm += 5
                    self.bpm_keys[1] = 0
                    self.loop_station.set_bpm(self.bpm)

                if keyboard.is_pressed("b"):
                    self.drum_key = 1

                if not keyboard.is_pressed("b") and self.drum_key == 1:
                    self.drum_key = 0
                    if self.base_drum_track:
                        self.base_drum_track = False
                        self.notes_selected[len(self.notes_selected) - 1] = [[], [], [], [], [], [], [], []]
                        self.loop_station.set_tones(self.notes_selected)
                    else:
                        self.base_drum_track = True
                        self.notes_selected[len(self.notes_selected) - 1] = [[2], [3], [2], [3], [2],
                                                                             [3], [2],
                                                                             [3]]
                        self.loop_station.set_tones(self.notes_selected)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()

    def undistort(self, img):
        DIM = (1280, 720)
        K = np.array(
            [[514.5414424836712, 0.0, 653.4964962297564], [0.0, 512.971045355613, 349.06466799474686],
             [0.0, 0.0, 1.0]])
        D = np.array(
            [[-0.055518374762623585], [-0.09175821441414458], [0.03752987112976255], [-0.011492161564419939]])

        # img = cv2.imread(img_path)
        h, w = img.shape[:2]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img

    def draw_lines(self, img):
        if self.outline:
            cv2.line(img, (self.score_xpos, self.scores_y_pos[1]), (self.score_xpos, self.scores_y_pos[-2]),
                     (255, 255, 255), 10)
            cv2.line(img, (img.shape[1] - self.score_xpos, self.scores_y_pos[1]),
                     (img.shape[1] - self.score_xpos, self.scores_y_pos[-2]), (255, 255, 255), 10)

        for i in range(1, len(self.scores_y_pos) - 1):
            h = self.scores_y_pos[i]
            if self.outline:
                cv2.line(img, (self.score_xpos, h), (img.shape[1] - self.score_xpos, h), (255, 255, 255), 6)
            cv2.line(img, (self.score_xpos, h), (img.shape[1] - self.score_xpos, h), (0, 0, 0), 3)

        cv2.line(img, (self.score_xpos, self.scores_y_pos[1]), (self.score_xpos, self.scores_y_pos[-2]), (0, 0, 0), 7)
        cv2.line(img, (img.shape[1] - self.score_xpos, self.scores_y_pos[1]),
                 (img.shape[1] - self.score_xpos, self.scores_y_pos[-2]), (0, 0, 0), 7)
        return img

    def draw_notes(self, img):
        # 0: face on note
        # 1: face on note
        # 2: countdown to select
        # 3: selected
        for index, instrument in enumerate(self.notes):
            for note in instrument:
                pos = (int(self.score_xpos + self.score_width - self.section_width * (note.beat + 0.5)), note.y_pos)
                if index == self.current_instrument:
                    if note.status == 1:
                        cv2.circle(img, pos, int(self.score_spacing / 2), note.status_1_color, 3)
                    if note.status == 2:
                        cv2.ellipse(img, pos, (int(self.score_spacing / 2), int(self.score_spacing / 2)), 0, 0,
                                    self.note_radius, note.status_2_color, 3)
                    if note.status == 3:
                        cv2.circle(img, pos, int(self.score_spacing / 2), note.status_3_color, -1)
                elif self.show_other_tracks:
                    if note.status == 3:
                        overlay = img.copy()
                        cv2.circle(overlay, pos, int(self.score_spacing / 2), note.status_3_color, -1)

                        alpha = 0.4  # Transparency factor.
                        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return img

    def check_pos_in_note(self, pos):
        return_value = False
        for note in self.notes[self.current_instrument]:
            if self.bigger_selection_space:
                x_pos = int(self.score_xpos + self.score_width - self.section_width * (note.beat))
                if x_pos - self.section_width < pos[0] < x_pos and note.y_pos - self.score_spacing / 2 < pos[
                    1] < note.y_pos + self.score_spacing / 2:
                    return_value = note.y_pos, note.beat
                else:
                    if note.status != 3:
                        if note.status != 0:
                            print("not selected", note.beat)
                        note.status = 0
            else:
                x_pos = int(self.score_xpos + self.score_width - self.section_width * (note.beat + 0.5))
                radius = int(self.score_spacing / 2)
                if math.sqrt((pos[0] - x_pos) ** 2 + (pos[1] - note.y_pos) ** 2) <= radius:
                    return_value = note.y_pos, note.beat
                else:
                    if note.status != 3:
                        note.status = 0
        return return_value

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

            self.head_color = (int(dominant[0]), int(dominant[1]), int(dominant[2]))
            if (self.head_color[2] * 2 + self.head_color[0] + self.head_color[1] * 3) / 6 / 255 < 0.25:
                self.head_outline_color = (255, 255, 255)
            else:
                self.head_outline_color = (0, 0, 0)

    def draw_user(self, image, results):

        # draw circle for face with nose in the middle
        image_height, image_width, _ = image.shape
        x_nose = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.NOSE].x * image_width
        y_nose = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.NOSE].y * image_height

        x_ear_l = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_EAR].x * image_width
        y_ear_l = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_EAR].y * image_height
        x_ear_r = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_EAR].x * image_width
        y_ear_r = results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.RIGHT_EAR].y * image_height

        radius_head = int(((x_ear_r - x_ear_l) ** 2 + (y_ear_r - y_ear_l) ** 2) ** 0.5 * 0.65)
        if self.outline:
            cv2.circle(image, (int(x_nose), int(y_nose)), int(radius_head), self.head_outline_color, 8)

        cv2.circle(image, (int(x_nose), int(y_nose)), int(radius_head), self.head_color, 6)

        return image

        """self.mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            self.mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles
            .get_default_face_mesh_contours_style())"""

        '''self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles
            .get_default_pose_landmarks_style())'''


detection = BodyDetection()
detection.webcam_analysis()
