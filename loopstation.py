import time
import pygame
import os
import rtmidi2


class LoopStation:
    def __init__(self, amount_beats):
        self.bpm = 90
        self.len_beat = 60 / self.bpm
        self.len_music = 120
        self.amount_beats = amount_beats

        self.channels = []
        self.notes = []
        self.velocities = []

        # array for each beat in one 8 count with what tone(s) to play
        self.tones = []
        for i in range(self.amount_beats):
            self.tones.append([])
        # print("tones", self.tones)

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
            if count > self.amount_beats:
                count = 1

        # self.midiout.send_noteoff(0, 60)

        self.channels = []
        self.notes = []
        self.velocities = []
        for i in range(len(self.tones)):
            for note in self.tones[i][count - 1]:
                self.channels.append(i)
                self.notes.append(15 * note + 30)
                self.velocities.append(100)

        self.midiout.send_noteon_many(self.channels, self.notes, self.velocities)
        print("message sent", self.channels)

        return count
