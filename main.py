import cv2
from Visualizations import Visualizations
from VideoAnalysis import VideoAnalysis
from Music import Music
import mediapipe as mp
import keyboard


class Main:
    def __init__(self):
        self.instruments = ["Bass", "Melody", "Drums"]
        self.current_instrument = 0
        # bass, melody
        self.rhythms = [[0.125] * 24, [0.125] * 24, [0.0625] * 48]
        self.notes_selected = [[-1] * len(self.rhythms[0]), [-1] * len(self.rhythms[1]), [-1] * len(self.rhythms[2])]
        # self.notes_selected[0][3] = 4

        self.screen_height = 900

        self.port_A_to_P = 0
        self.port_P_to_Q = 0

        self.score_width = 1400
        self.height_start = 220
        self.height_stop = self.screen_height - 300

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

                self.video_analysis.analyze(image, holistic)

                self.visualizations.draw_score(image)
                self.visualizations.draw_notes(image, self.notes_selected, self.current_instrument)
                self.visualizations.draw_moving_line(image)

                if self.video_analysis.human_detected():
                    self.video_analysis.update_head_pos(image_height, image_width)

                    head_over_note = self.video_analysis.check_head_in_note(self.current_instrument,
                                                                            self.visualizations.notes_margin_left,
                                                                            self.visualizations.notes_y_pos)
                    if head_over_note:
                        self.notes_selected[self.current_instrument][head_over_note[0]] = head_over_note[1]


                counter = self.music.react_to_messages(self.notes_selected)
                if counter == 0 and self.previous_counter != 0:
                    self.visualizations.set_track_len()
                    self.previous_counter = 0
                if counter:
                    self.previous_counter = counter


                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, self.instruments[self.current_instrument], (10, 70), font, 2.5,
                            (0, 0, 0), 10, cv2.LINE_AA)
                cv2.putText(image, self.instruments[self.current_instrument], (10, 70), font, 2.5,
                            (255, 255, 255), 3, cv2.LINE_AA)

                cv2.imshow(self.window_name, image)

                for i in range(1, 4):
                    if keyboard.is_pressed(i + 1):
                        self.current_instrument = i - 1
                        print("current", self.current_instrument)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()


main = Main()
main.start_camera()
