import cv2
import numpy as np
import time


class Visualizations:

    def __init__(self, score_width, height_start, height_stop, rhythms, amount_instruments):
        self.score_width = score_width
        self.height_start = height_start
        self.score_xpos = 0
        self.score_spacing = (height_stop - height_start) / 6
        # y position of score line including start and end point(which do not have lines drawn on them)
        self.scores_y_pos = [height_start]
        # y position of notes with also notes on lines
        self.notes_y_pos = [height_start]
        self.rhythms = rhythms

        self.len_track = 0
        self.last_track_time = time.time()

        for i in range(1, 6):
            h = int(height_start + i * self.score_spacing)
            self.scores_y_pos.append(h)

            self.notes_y_pos.append(h - self.score_spacing / 2)
            self.notes_y_pos.append(h)
        self.scores_y_pos.append(height_stop)
        self.notes_y_pos.append(height_stop)

        full_note_width = self.score_width / sum(rhythms[0])

        self.notes_widths = []
        for i in range(amount_instruments):
            self.notes_widths.append([])
            for note_len in rhythms[i]:
                self.notes_widths[i].append(note_len * full_note_width)

        self.notes_margin_left = [[0], [0], [0]]
        for i in range(amount_instruments):
            width = 0
            for note_len in rhythms[i]:
                width += note_len * full_note_width
                self.notes_margin_left[i].append(width)

    def set_track_len(self):
        self.len_track = time.time() - self.last_track_time
        self.last_track_time = time.time()

    def draw_moving_line(self, img):
        # draw moving time line
        if self.len_track != 0:
            line_x_pos = int(self.score_xpos + self.score_width * (time.time() - self.last_track_time) / self.len_track)
            cv2.line(img, (line_x_pos, 0), (line_x_pos, img.shape[1]), (0, 0, 0), 2)

    def draw_score(self, img):
        cv2.line(img, (self.score_xpos, self.scores_y_pos[1]), (self.score_xpos, self.scores_y_pos[-2]),
                 (255, 255, 255), 10)
        cv2.line(img, (img.shape[1] - self.score_xpos, self.scores_y_pos[1]),
                 (img.shape[1] - self.score_xpos, self.scores_y_pos[-2]), (255, 255, 255), 10)

        for i in range(1, len(self.scores_y_pos) - 1):
            h = self.scores_y_pos[i]
            cv2.line(img, (self.score_xpos, h), (img.shape[1] - self.score_xpos, h), (255, 255, 255), 6)
            cv2.line(img, (self.score_xpos, h), (img.shape[1] - self.score_xpos, h), (0, 0, 0), 3)

        cv2.line(img, (self.score_xpos, self.scores_y_pos[1]), (self.score_xpos, self.scores_y_pos[-2]), (0, 0, 0), 7)
        cv2.line(img, (img.shape[1] - self.score_xpos, self.scores_y_pos[1]),
                 (img.shape[1] - self.score_xpos, self.scores_y_pos[-2]), (0, 0, 0), 7)

        """for i in self.notes_margin_left[0]:
            cv2.line(img, (int(i), self.scores_y_pos[0]),
                     (int(i), self.scores_y_pos[-1]), (0, 0, 0), 3)"""
        return img

    def draw_notes(self, img, notes_selected, current_instrument):
        for index_instr, instrument in enumerate(notes_selected):
            for column, row in enumerate(instrument):
                if row != -1:
                    x_pos_center = int(
                        self.notes_margin_left[index_instr][column] + self.notes_widths[index_instr][column] / 2)
                    y_pos_center = int(self.notes_y_pos[int(row * 2)] + self.score_spacing / 2)
                    if index_instr == 0:
                        if self.rhythms[0][column] == 0.125:
                            cv2.circle(img, [x_pos_center, y_pos_center], int(self.score_spacing / 2), (0, 200, 0), -1)

                        elif self.rhythms[0][column] == 0.25:
                            cv2.circle(img, [x_pos_center, y_pos_center], int(self.score_spacing / 2), (0, 200, 0), 3)

                    if index_instr == 1:
                        if self.rhythms[1][column] == 0.125:
                            cv2.circle(img, [x_pos_center, y_pos_center], int(self.score_spacing / 2), (200, 0, 0), -1)

                        elif self.rhythms[1][column] == 0.25:
                            cv2.circle(img, [x_pos_center, y_pos_center], int(self.score_spacing / 2), (200, 0, 0), 3)
                    if index_instr == 2:
                        if column != 0:
                            x_pos_center_start = int(
                                self.notes_margin_left[index_instr][column - 1] + self.notes_widths[index_instr][
                                    column - 1] / 2)
                            y_pos_center_start = int(
                                self.notes_y_pos[int(instrument[column - 1] * 2)] + self.score_spacing / 2)
                            cv2.line(img, [x_pos_center_start, y_pos_center_start], [x_pos_center, y_pos_center],
                                     (0, 0, 200), 3)

    def draw_user(self, img):
        pass

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
