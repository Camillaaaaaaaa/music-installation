from AbletonCommunication import AbletonCommunication


class Music:
    def __init__(self):
        self.ableton = AbletonCommunication()
        self.melody_value_mapping = {5: 62, 4.5: 63, 4: 64, 3.5: 65, 3: 66, 2.5: 67, 2: 68, 1.5: 69, 1: 70, 0.5: 71,
                                     0: 72}
        self.melody_counter = 0
        self.bass_counter = 0
        self.drums_counter = 0

    def react_to_messages(self, selected_notes):
        message = self.ableton.get_message()
        # [messagetype+channel, value1, value2]
        # 176=CC
        if message:
            # melody
            if message[0] == 144 + 15:
                self.play_melody(selected_notes[1])
                self.melody_counter += 1
                # bass
                self.play_bass(selected_notes[0])
                self.bass_counter += 1

                # drums
                self.play_drums(selected_notes[0])
                self.drums_counter += 1

                if message[1] == 43:
                    self.bass_counter = 0
                    self.melody_counter = 0
                    self.drums_counter = 0

            # drums also on node off message
            if message[0] == 128 + 15:
                # drums
                self.play_drums(selected_notes[0])
                self.drums_counter += 1

            return self.melody_counter

    def play_drums(self, selected_notes):
        if selected_notes[self.melody_counter] != -1:
            value = self.melody_value_mapping[selected_notes[self.melody_counter]]
            # self.ableton.send_message("CC", 0, 76, value)
        else:
            pass
            # self.ableton.send_message("CC", 0, 76, 0)

    def play_melody(self, selected_notes):
        if selected_notes[self.melody_counter] != -1:
            value = self.melody_value_mapping[selected_notes[self.melody_counter]]
            self.ableton.send_message("CC", 0, 76, value)
        else:
            self.ableton.send_message("CC", 0, 76, 0)

    def play_bass(self, selected_notes):
        if selected_notes[self.bass_counter] != -1:
            value = self.melody_value_mapping[selected_notes[self.bass_counter]]
            self.ableton.send_message("CC", 1, 76, value)
        else:
            self.ableton.send_message("CC", 1, 76, 0)
