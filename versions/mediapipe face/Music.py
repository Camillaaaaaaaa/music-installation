from AbletonCommunication import AbletonCommunication


class Music:
    def __init__(self):
        self.ableton = AbletonCommunication()
        self.melody_value_mapping = {5: 54, 4.5: 59, 4: 64, 3.5: 68, 3: 71, 2.5: 76, 2: 81, 1.5: 84, 1: 89, 0.5: 94,
                                     0: 100}

        self.drums_value_mapping = {}
        for i in range(0, 11):
            i = i / 2
            self.drums_value_mapping[i] = [125 - i * 9, 125 - 5 * 9 + i * 9]

        self.melody_counter = 0
        self.bass_counter = 0
        self.drums_counter = 0

    def react_to_messages(self, selected_notes):
        message = self.ableton.get_message()
        # [messagetype+channel, value1, value2]
        # 176=CC
        #print(message)
        if message and message[0]:
            self.bass_counter = message[1]
            self.melody_counter = message[1]
            self.drums_counter = message[1]
            # melody
            self.play_melody(selected_notes[2])
            #self.melody_counter += 1
            # bass
            self.play_bass(selected_notes[1])
            #self.bass_counter += 1

            # drums
            self.play_drums(selected_notes[0])
            #self.drums_counter += 1

            if message[1]==23:
                print("hi")
                self.ableton.reopen_midi_in()


            """#print(message[1])
            #if message[1] == 43 or self.melody_counter >= len(selected_notes[2]):
            if self.melody_counter >= len(selected_notes[2]):
                #print(message[1])
                self.bass_counter = 0
                self.melody_counter = 0
                self.drums_counter = 0
                self.ableton.reopen_midi_in()"""

        return self.melody_counter

    def play_drums(self, selected_notes):
        if self.drums_counter < len(selected_notes) and selected_notes[self.drums_counter] != -1:
            value = self.drums_value_mapping[selected_notes[self.drums_counter]]
            self.ableton.send_message("CC", 2, 76, value[1])
            self.ableton.send_message("CC", 3, 76, value[0])
        else:
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
            value = self.melody_value_mapping[selected_notes[self.bass_counter]]
            self.ableton.send_message("CC", 1, 76, value)

            # turn on bass track
            self.ableton.send_message("CC", 5, 76, 100)
        else:
            # turn off bass track
            self.ableton.send_message("CC", 5, 76, 0)

    def set_filter(self, index, on):
        if on:
            self.ableton.send_message("CC", 6 + index, 76, 100)
        else:
            self.ableton.send_message("CC", 6 + index, 76, 0)
