import time
import pygame
import os
import rtmidi2


class LoopStation:
    def __init__(self):
        self.bpm = 90
        self.len_beat = 60 / self.bpm
        self.len_music = 120

        # array for each beat in one 8 count with what tone(s) to play
        self.tones = [[], [], [], [], [], [], [], []]

        self.midiout = rtmidi2.MidiOut()
        ports = rtmidi2.get_out_ports()
        print(ports)
        self.midiout.open_port(1)

    def set_tones(self, new_tones):
        self.tones = new_tones

    def play_tone(self, count):
        if count == 1:
            # self.beep.play()
            count += 1
        else:
            # self.bop.play()
            count += 1
            if count > 8:
                count = 1

        #self.midiout.send_noteoff(0, 60)

        if self.tones[count - 1]:
            for tone in self.tones[count - 1]:
                print(15*tone+30)
                self.midiout.send_noteon(0, 80-10*tone, 100)

        return count

    def play_tones(self):
        start_time = time.time()
        last_time = time.time()
        count = 1

        while time.time() < start_time + self.len_music:
            # one beat is over
            if time.time() >= last_time + self.len_beat:
                print("beat")
                pygame.mixer.stop()
                if self.tones[count - 1]:
                    for tone in self.tones[count - 1]:
                        pass
                        self.allTones[tone].play()

                if count == 1:
                    last_time = time.time()
                    self.beep.play()
                    count += 1
                else:
                    last_time = time.time()
                    self.bop.play()
                    count += 1
                    if count > 8:
                        count = 1
