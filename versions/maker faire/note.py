class Note:

    def __init__(self, score_pos, beat):
        self.score_pos = score_pos
        # 0: face on note
        # 1: face on note
        # 2: countdown to select
        # 3: selected
        self.status = 0
        self.y_pos = 0
        self.beat = beat
        self.status_1_color = (100, 100, 100)
        self.status_2_color = (99, 235, 255)
        self.status_3_color = (20, 40, 186)
