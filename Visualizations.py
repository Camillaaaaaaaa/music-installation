import cv2
import numpy as np


class Visualizations:

    def __init__(self, height_start, height_stop, rhythms, amount_instruments, score_offset_x):
        self.score_width = 0
        self.height_start = height_start
        # offset to left of screen
        self.score_xpos = score_offset_x
        self.score_spacing = (height_stop - height_start) / 6
        # y position of score line including start and end point(which do not have lines drawn on them)
        self.scores_y_pos = [height_start]
        # y position of notes with also notes on lines
        self.notes_y_pos = [height_start]
        self.rhythms = rhythms
        self.white = (227, 224, 228)
        self.amount_instruments = amount_instruments

        self.drums_size = 1

        self.beat = 0

        # y positions of music score lines and note rows
        for i in range(1, 6):
            h = int(height_start + i * self.score_spacing)
            self.scores_y_pos.append(h)

            self.notes_y_pos.append(h - self.score_spacing / 2)
            self.notes_y_pos.append(h)
        self.scores_y_pos.append(height_stop)
        self.notes_y_pos.append(height_stop)

        self.notes_widths = []

        self.notes_margin_left = [[self.score_xpos], [self.score_xpos], [self.score_xpos]]

        self.mask1 = []
        self.mask2 = []

    def set_width(self, width):
        # set width of score and x position of notes when screen width is known
        self.score_width = width
        full_note_width = self.score_width / sum(self.rhythms[0])

        # widths of note columns
        self.notes_widths = []
        for i in range(self.amount_instruments):
            self.notes_widths.append([])
            for note_len in self.rhythms[i]:
                self.notes_widths[i].append(note_len * full_note_width)

        # offset of each column of notes
        self.notes_margin_left = [[self.score_xpos], [self.score_xpos], [self.score_xpos]]
        for i in range(self.amount_instruments):
            width = self.score_xpos
            for note_len in self.rhythms[i]:
                width += note_len * full_note_width
                self.notes_margin_left[i].append(width)

    def draw_background(self, img, type):
        if type == 1:
            # draw opaque background behind music score
            for i in range(0, 100, 10):
                overlay = img.copy()
                cv2.rectangle(overlay, (self.score_xpos - i - 15, 0), (self.score_xpos, overlay.shape[0]), self.white, -1)

                alpha = 0.2  # Transparency factor.
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            cv2.rectangle(img, (self.score_xpos - 15, 0), (img.shape[1], img.shape[0]), self.white, -1)
            return img
        elif type == 2:
            # add filter gradually behind music score

            # convolution filter on image to create outlines
            emboss_kernel = 3 * np.array([[-1, -1, -1],
                                          [-1, 8, -1],
                                          [-1, -1, -1]])
            emboss_img = cv2.filter2D(src=img, ddepth=-1, kernel=emboss_kernel)
            emboss_img = cv2.bitwise_not(emboss_img)

            # https://stackoverflow.com/questions/58444510/creating-a-linear-gradient-mask-using-opencv-or
            # Generate blend masks, here: linear, horizontal fading from 1 to 0 and from 0 to 1
            # array1 = [1] * int(img.shape[1] * 0.1)
            if self.mask1 == []:
                array2 = np.linspace(0.7, 0, int(img.shape[1] * 0.7))
                array3 = [0] * int(img.shape[1] * 0.3)

                array = [*array2, *array3]
                self.mask1 = np.repeat(np.tile(array, (img.shape[0], 1))[:, :, np.newaxis], 3, axis=2)
                array = np.flip(array)
                self.mask2 = np.repeat(np.tile(array, (emboss_img.shape[0], 1))[:, :, np.newaxis], 3, axis=2)

            # cv2.imshow("mask",mask2)
            # Generate output by linear blending
            final = np.uint8(img * self.mask1 + emboss_img * self.mask2)

            return final

    def draw_moving_line(self, img):
        # draw moving time line that shows current beat
        x_pos_start = int(self.notes_margin_left[0][self.beat])
        x_pos_end = int(self.notes_margin_left[0][self.beat + 1])

        overlay = img.copy()
        cv2.rectangle(overlay, (x_pos_start, 0), (x_pos_end, img.shape[0]), (0, 0, 0), -1)

        alpha = 0.4  # Transparency factor.
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return img

    def draw_score(self, img):
        # side lines
        cv2.rectangle(img, (self.score_xpos - 14, self.scores_y_pos[1] - 4),
                      (self.score_xpos + 4, self.scores_y_pos[-2] + 4),
                      self.white, -1)
        cv2.rectangle(img, (self.score_xpos + self.score_width - 4, self.scores_y_pos[1] - 4),
                      (self.score_xpos + self.score_width + 14, self.scores_y_pos[-2] + 4),
                      self.white, -1)

        # horizontal lines
        for i in range(1, len(self.scores_y_pos) - 1):
            h = self.scores_y_pos[i]
            cv2.line(img, (self.score_xpos, h), (self.score_xpos + self.score_width, h), self.white,
                     8)
            cv2.line(img, (self.score_xpos, h), (self.score_xpos + self.score_width, h), (0, 0, 0), 3)


        # side lines
        cv2.rectangle(img, (self.score_xpos - 10, self.scores_y_pos[1] - 2),
                      (self.score_xpos, self.scores_y_pos[-2] + 2), (0, 0, 0), -1)
        cv2.rectangle(img, (self.score_xpos + self.score_width - 2, self.scores_y_pos[1] - 2),
                      (self.score_xpos + self.score_width + 10, self.scores_y_pos[-2] + 2), (0, 0, 0),
                      -1)
        return img

    def draw_notes(self, img, notes_selected, colors, shirt_colors, detect_shirt_color):
        for index_instr, instrument in enumerate(notes_selected):
            for column, row in enumerate(instrument):
                if row != -1:
                    # calculate note position from row and column
                    x_pos_center = int(
                        self.notes_margin_left[index_instr][column] + self.notes_widths[index_instr][column] / 2)
                    y_pos_center = int(self.notes_y_pos[int(row * 2)] + self.score_spacing / 2)

                    # if its a drum note, draw "pulsing" note for each bass drum sound
                    if index_instr == 0:
                        overlay = img.copy()

                        cv2.circle(overlay, [x_pos_center, y_pos_center], int(self.score_spacing / 2 * self.drums_size),
                                   colors[index_instr], -1)

                        alpha = 0.6  # Transparency factor.
                        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                    cv2.circle(img, [x_pos_center, y_pos_center], int(self.score_spacing / 2),
                               colors[index_instr], -1)

                    if detect_shirt_color:
                        cv2.circle(img, [x_pos_center, y_pos_center], int(self.score_spacing / 2) - 5,
                                   shirt_colors[index_instr], 5)
        return img

    def draw_user(self, img, pos, color, size, shirt_color, detect_shirt_color):
        if detect_shirt_color:
            cv2.circle(img, (int(pos[0]), int(pos[1])), int(size), (0, 0, 0), 30)
            cv2.circle(img, (int(pos[0]), int(pos[1])), int(size), shirt_color, 25)
        else:
            cv2.circle(img, (int(pos[0]), int(pos[1])), int(size), (0, 0, 0), 30)
            cv2.circle(img, (int(pos[0]), int(pos[1])), int(size), color, 25)

    def write_instruments(self, image, instruments, instruments_color, current_instrument):
        # write text at the top of the screen
        font = cv2.FONT_HERSHEY_COMPLEX
        for index, instrument in enumerate(instruments):
            if index == current_instrument:
                cv2.putText(image, instrument, (self.score_xpos - 80 + 320 * index, 70), font, 2.5,
                            (0, 0, 0), 20, cv2.LINE_AA)
                cv2.putText(image, instrument, (self.score_xpos - 80 + 320 * index, 70), font, 2.5,
                            self.white, 13, cv2.LINE_AA)
                cv2.putText(image, instrument, (self.score_xpos - 80 + 320 * index, 70), font, 2.5,
                            instruments_color[index], 5, cv2.LINE_AA)
            else:
                cv2.putText(image, instrument, (self.score_xpos - 80 + 320 * index, 70), font, 2.5,
                            (0, 0, 0), 10, cv2.LINE_AA)
                cv2.putText(image, instrument, (self.score_xpos - 80 + 320 * index, 70), font, 2.5,
                            self.white, 6, cv2.LINE_AA)
                cv2.putText(image, instrument, (self.score_xpos - 80 + 320 * index, 70), font, 2.5,
                            instruments_color[index], 2, cv2.LINE_AA)

    def draw_control_img(self, instruments, instruments_color):
        img = np.zeros((250, 650, 3), np.uint8)
        img[:, :] = self.white

        font = cv2.FONT_HERSHEY_COMPLEX

        cv2.putText(img, "on/off  change  reset", (10, 30), font, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        for index, instrument in enumerate(instruments):
            text = str(index + 1) + "   " + str(index + 4) + "   " + str(
                index + 7) + "   " + instrument
            cv2.putText(img, text,
                        (10, 80 + 70 * index),
                        font, 1.5,
                        (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(img, text,
                        (10, 80 + 70 * index),
                        font, 1.5,
                        instruments_color[index], 2, cv2.LINE_AA)

        cv2.imshow("control", img)

    def undistort(self, img):
        # https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
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
