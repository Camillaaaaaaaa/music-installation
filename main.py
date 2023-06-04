import cv2
from Visualizations import Visualizations
from VideoAnalysis import VideoAnalysis
from Music import Music
import mediapipe as mp
import keyboard


class Main:
    def __init__(self):
        self.instruments = ["Drums", "Bass", "Melody"]
        self.current_instrument = 0
        self.instruments_color = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
        self.instruments_main_color = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
        # bass, melody
        self.rhythms = [[0.125] * 24, [0.125] * 24, [0.125] * 24]
        self.notes_selected = [[-1] * len(self.rhythms[0]), [-1] * len(self.rhythms[1]), [-1] * len(self.rhythms[2])]
        # self.notes_selected[0][3] = 4

        self.screen_height = 900

        self.port_A_to_P = 0
        self.port_P_to_Q = 0

        self.score_width = 1400
        self.height_start = 220
        self.height_stop = self.screen_height - 300

        self.frame_counter = 25

        self.music_change = [False, False, False]
        self.key_pressed = [False, False, False]

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic

        self.visualizations = Visualizations(self.score_width, self.height_start, self.height_stop, self.rhythms,
                                             len(self.instruments))
        self.video_analysis = VideoAnalysis(self.mp_holistic)

        self.music = Music()

        self.window_name = "loop station"

        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.previous_counter = 1
        self.set_track_len_amount = 0

    def start_camera(self):
        cap = cv2.VideoCapture(0)
        with self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                success, image = cap.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                image = cv2.resize(image,
                                   (int(self.screen_height / image.shape[0] * image.shape[1]), self.screen_height),
                                   interpolation=cv2.INTER_LINEAR)

                image_height, image_width, _ = image.shape

                image = cv2.flip(image, 1)

                # image = self.undistort(image)

                # set dimensions of score
                if self.visualizations.score_xpos == 0:
                    self.visualizations.score_xpos = int((image_width - self.score_width) / 2)
                    for i in range(len(self.visualizations.notes_margin_left)):
                        for a in range(len(self.visualizations.notes_margin_left[i])):
                            self.visualizations.notes_margin_left[i][a] += self.visualizations.score_xpos

                # get all people in the image
                people = self.video_analysis.find_users(image)
                # print([(p[1], p[2]) for p in people])
                # sort after who is in front / x position
                people.sort(key=lambda row: row[1] + row[0].shape[0] / 2, reverse=True)
                # print("sorted", [(p[1], p[2]) for p in people])

                for index, person in enumerate(people):
                    results = self.video_analysis.analyze(person[0], holistic)
                    frame_height, frame_width, _ = person[0].shape

                    instrument = (self.current_instrument + index) % len(self.instruments)

                    if results.pose_landmarks:
                        head_pos = self.video_analysis.update_head_pos(frame_height, frame_width, person[1], person[2],
                                                                       results)

                        head_over_note = self.video_analysis.check_head_in_note(
                            instrument, self.visualizations.notes_margin_left, self.visualizations.notes_y_pos,
                            head_pos)

                        if head_over_note:
                            self.notes_selected[instrument][head_over_note[0]] = head_over_note[1]

                        if self.frame_counter >= 15:
                            self.frame_counter = 0
                            shirt_color = self.video_analysis.detect_shirt_color(image, results)
                            self.instruments_color[instrument] = shirt_color
                        self.frame_counter += 1

                        head_width = self.video_analysis.get_face_size(frame_height, frame_width, results)
                        self.visualizations.draw_user(image, head_pos, self.instruments_color[instrument], head_width)

                self.visualizations.draw_score(image)
                self.visualizations.draw_notes(image, self.notes_selected, self.instruments_color,self.instruments_main_color)
                image = self.visualizations.draw_moving_line(image)

                self.visualizations.write_instruments(image, self.instruments, self.instruments_color,
                                                      self.current_instrument,self.instruments_main_color)

                counter = self.music.react_to_messages(self.notes_selected)
                self.visualizations.beat = counter
                """if counter == 0 and self.previous_counter != 0:
                    self.visualizations.set_track_len()
                    self.previous_counter = 0
                if counter:
                    self.previous_counter = counter"""

                cv2.imshow(self.window_name, image)

                self.visualizations.draw_control_img(self.instruments, self.instruments_color, self.music_change, self.instruments_main_color)

                self.key_handler()

                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()

    def key_handler(self):
        for i in range(1, 4):
            if keyboard.is_pressed(i + 1):
                self.current_instrument = i - 1
                self.notes_selected[i - 1] = [-1] * len(self.rhythms[i - 1])
                print("current", self.current_instrument)

        if keyboard.is_pressed("q") and not self.key_pressed[0]:
            self.music_change[0] = not self.music_change[0]
            self.music.set_filter(0, self.music_change[0])
            self.key_pressed[0] = True

        if not keyboard.is_pressed("q"):
            self.key_pressed[0] = False

        if keyboard.is_pressed("w") and not self.key_pressed[1]:
            self.music_change[1] = not self.music_change[1]
            self.music.set_filter(1, self.music_change[1])
            self.key_pressed[1] = True
        if not keyboard.is_pressed("w"):
            self.key_pressed[1] = False

        if keyboard.is_pressed("e") and not self.key_pressed[2]:
            self.music_change[2] = not self.music_change[2]
            self.music.set_filter(2, self.music_change[2])
            self.key_pressed[2] = True

        if not keyboard.is_pressed("e"):
            self.key_pressed[2] = False


main = Main()
main.start_camera()
