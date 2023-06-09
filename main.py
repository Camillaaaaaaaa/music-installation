import cv2
import keyboard
import mediapipe as mp

from Music import Music
from VideoAnalysis import VideoAnalysis
from Visualizations import Visualizations


class Main:
    def __init__(self):
        self.show_second_screen = False
        self.detect_shirt_color = False
        self.instruments_shirt_color = [(197, 95, 89), (89, 174, 197), (89, 97, 197)]

        self.instruments = ["Drums", "Bass", "Melody"]
        self.current_instrument = 0
        self.possible_colors = [[(196, 94, 88), (112, 39, 35)],
                                [(63, 140, 112), (58, 89, 66)],
                                [(68, 91, 187), (29, 43, 104)]]
        self.current_color = [0, 0, 0]
        self.instruments_color = []
        for index, i in enumerate(self.current_color):
            self.instruments_color.append(self.possible_colors[index][i])

        self.rhythms = [[0.125] * 24, [0.125] * 24, [0.125] * 24]
        self.notes_selected = [[-1] * len(self.rhythms[0]), [-1] * len(self.rhythms[1]), [-1] * len(self.rhythms[2])]

        self.screen_height = 900

        self.score_width = 900
        self.height_start = 250
        self.height_stop = self.screen_height - 300
        self.score_offset_x = 350

        self.frame_counter = 25
        self.analysis_counter = 2

        self.music_change = [False, False, False]
        self.key_pressed = [False, False, False]

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic

        self.visualizations = Visualizations(self.height_start, self.height_stop, self.rhythms,
                                             len(self.instruments), self.score_offset_x)
        self.video_analysis = VideoAnalysis(self.mp_holistic)

        self.music = Music()

        for i in range(3):
            self.music.set_filter(i, self.music_change[i])

        self.window_name = "loop station"

        cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.previous_counter = 1
        self.previous_head = []
        self.multiple_people = True

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
                # create one image for analysis and one for the visualizations
                image_visualization = image.copy()

                # image = self.undistort(image)

                # set score witdh if it was not set yet
                if self.visualizations.score_width == 0:
                    self.visualizations.set_width(int(image.shape[1] - self.score_offset_x - 50))

                image_visualization = self.visualizations.draw_background(image_visualization, 2)

                # analyze only every second frame to increase performance
                if self.analysis_counter == 2 or not self.multiple_people:
                    self.analysis_counter = 0
                    self.previous_head = []
                    if self.multiple_people:
                        # get all people in the image
                        people = self.video_analysis.find_users(image)
                        # sort after who is in front / x position
                        people.sort(key=lambda row: row[1] + row[0].shape[0] / 2, reverse=True)
                    else:
                        people = [(image, 0, 0)]

                    for index, person in enumerate(people):
                        # analyze image of person
                        results = self.video_analysis.analyze(person[0], holistic)
                        frame_height, frame_width, _ = person[0].shape

                        # instruments increase for each user when multiple users are in frame
                        instrument = (self.current_instrument + index) % len(self.instruments)

                        if results.pose_landmarks:
                            head_pos = self.video_analysis.update_head_pos(frame_height, frame_width, person[1],
                                                                           person[2],
                                                                           results)

                            head_over_note = self.video_analysis.check_head_in_note(
                                instrument, self.visualizations.notes_margin_left, self.visualizations.notes_y_pos,
                                head_pos)

                            if head_over_note:
                                self.notes_selected[instrument][head_over_note[0]] = head_over_note[1]

                            if self.detect_shirt_color:
                                # detect shirt color every 15 frames
                                if self.frame_counter >= 15:
                                    self.frame_counter = 0
                                    shirt_color = self.video_analysis.detect_shirt_color(image, results)
                                    self.instruments_shirt_color[instrument] = shirt_color
                                self.frame_counter += 1

                            head_width = self.video_analysis.get_face_size(frame_height, frame_width, results)
                            self.previous_head.append([head_pos, self.instruments_color[instrument], head_width,
                                                       self.instruments_shirt_color[instrument]])
                            self.visualizations.draw_user(image_visualization, head_pos,
                                                          self.instruments_color[instrument],
                                                          head_width, self.instruments_shirt_color[instrument],
                                                          self.detect_shirt_color)

                else:
                    # if current frame is not analyzed, take head position from previous frame
                    for person in self.previous_head:
                        if len(person) == 4:
                            self.visualizations.draw_user(image_visualization, person[0], person[1], person[2],
                                                          person[3],
                                                          self.detect_shirt_color)
                self.analysis_counter += 1

                self.visualizations.draw_score(image_visualization)
                image_visualization = self.visualizations.draw_notes(image_visualization, self.notes_selected,
                                                                     self.instruments_color,
                                                                     self.instruments_shirt_color,
                                                                     self.detect_shirt_color)
                image_visualization = self.visualizations.draw_moving_line(image_visualization)

                if self.show_second_screen:
                    image_second=image_visualization.copy()
                    image_visualization = cv2.flip(image_visualization, 1)
                    self.visualizations.write_instruments(image_second, self.instruments, self.instruments_color,
                                                          self.current_instrument)


                self.visualizations.write_instruments(image_visualization, self.instruments, self.instruments_color,
                                                      self.current_instrument)

                counter = self.music.react_to_messages(self.notes_selected)
                # print(counter)
                self.visualizations.beat = counter

                # set drum size to create "pulsing" drum note
                if self.previous_counter != counter and counter % 2 == 1:
                    if self.notes_selected[0][counter] != -1:
                        self.visualizations.drums_size = 1.3

                self.previous_counter = counter

                if self.visualizations.drums_size > 1:
                    self.visualizations.drums_size -= 0.05

                #self.visualizations.draw_control_img(self.instruments, self.instruments_color)

                cv2.imshow(self.window_name, image_visualization)
                if self.show_second_screen:
                    cv2.imshow("mirrored image", image_second)


                self.key_handler()

                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()

    def key_handler(self):
        # change current instrument
        for i in range(1, 4):
            if keyboard.is_pressed(i + 1):
                self.current_instrument = i - 1

        # turn audio filters on/off
        for i in range(4, 7):
            index = (i - 1) % 3
            if keyboard.is_pressed(i + 1) and not self.key_pressed[index]:
                self.key_pressed[index] = True

                self.current_color[index] = (self.current_color[index] + 1) % len(self.possible_colors[index])
                self.instruments_color[index] = self.possible_colors[index][self.current_color[index]]
                self.music_change[index] = not self.music_change[index]
                self.music.set_filter(index, self.music_change[index])

            if not keyboard.is_pressed(i + 1):
                self.key_pressed[(i - 1) % 3] = False

        # reset tracks
        for i in range(7, 10):
            if keyboard.is_pressed(i + 1):
                self.notes_selected[(i - 1) % 3] = [-1] * len(self.rhythms[(i - 1) % 3])


main = Main()
main.start_camera()
