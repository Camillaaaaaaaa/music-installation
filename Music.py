from AbletonCommunication import AbletonCommunication


class Music:
    def __init__(self):
        self.ableton = AbletonCommunication()

        # mapping between note position (0-5) and midi value for the pitch modulation (0-127)
        self.melody_value_mapping = {5: 54, 4.5: 59, 4: 64, 3.5: 68, 3: 71, 2.5: 76, 2: 81, 1.5: 84, 1: 89, 0.5: 94,
                                     0: 100}

        self.bass_value_mapping = {5: 54, 4.5: 59, 4: 64, 3.5: 68, 3: 71, 2.5: 76, 2: 81, 1.5: 84, 1: 89, 0.5: 94,
                                     0: 100}

        # mapping between note position (0-5) and drum volumes for low and high drums(0-127)
        self.drums_value_mapping = {}
        for i in range(0, 11):
            i = i / 2
            self.drums_value_mapping[i] = [125 - i * 9, 125 - 5 * 9 + i * 9]

        # beat counter
        self.melody_counter = 0
        self.bass_counter = 0
        self.drums_counter = 0

    def react_to_messages(self, selected_notes):
        # message: new beat True/ false, beat counter
        message = self.ableton.get_message()

        if message and message[0]:
            self.bass_counter = message[1]
            self.melody_counter = message[1]
            self.drums_counter = message[1]
            # melody
            self.play_melody(selected_notes[2])
            # bass
            self.play_bass(selected_notes[1])

            # drums
            self.play_drums(selected_notes[0])

        return self.melody_counter

    def play_drums(self, selected_notes):
        if self.drums_counter < len(selected_notes) and selected_notes[self.drums_counter] != -1:
            value = self.drums_value_mapping[selected_notes[self.drums_counter]]
            # set volume on low and high drums
            self.ableton.send_message("CC", 2, 76, value[1])
            self.ableton.send_message("CC", 3, 76, value[0])
        else:
            # turn drums off if no note is selected for specific beat
            self.ableton.send_message("CC", 2, 76, 0)
            self.ableton.send_message("CC", 3, 76, 0)

    def play_melody(self, selected_notes):
        if selected_notes[self.melody_counter] != -1:
            # set note
            value = self.melody_value_mapping[selected_notes[self.melody_counter]]
            self.ableton.send_message("CC", 0, 76, value)

            # turn on melody track
            self.ableton.send_message("CC", 4, 76, 100)
        else:
            # turn off melody track
            self.ableton.send_message("CC", 4, 76, 0)

    def play_bass(self, selected_notes):
        if selected_notes[self.bass_counter] != -1:
            # set bass note
            value = self.bass_value_mapping[selected_notes[self.bass_counter]]
            self.ableton.send_message("CC", 1, 76, value)

            # turn on bass track
            self.ableton.send_message("CC", 5, 76, 100)
        else:
            # turn off bass track
            self.ableton.send_message("CC", 5, 76, 0)

    def set_filter(self, index, on):
        # turn specific audio filter on or off
        if on:
            self.ableton.send_message("CC", 6 + index, 76, 100)
        else:
            self.ableton.send_message("CC", 6 + index, 76, 0)
