import time
import pygame
import os
import rtmidi2


class LoopStation:
    def __init__(self, amount_beats):
        self.bpm = 110
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

        # instrument mappings

        # bass drum: C1, rim shot: C#1, Snare Drum: D1, Hand Clap: D#1, Snare Drum: G1, Ride:D2
        drums = {5: 36, 4: 37, 3: 38, 2: 39, 1: 43, 0: 50}

        # D2, F2, A2, C3, E3, G3
        piano = {5: 50, 4: 53, 3: 57, 2: 60, 1: 64, 0: 67}

        # D0, F0, A0, C1, E1, G1
        bass = {5: 26, 4: 29, 3: 33, 2: 36, 1: 40, 0: 43}

        #  D2, F2, A2, C3, E3, G3
        guitar = {5: 50, 4: 53, 3: 57, 2: 60, 1: 64, 0: 67}

        self.instruments = [drums, piano, bass, guitar]

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
                self.notes.append(self.instruments[i][note])
                self.velocities.append(100)

        self.midiout.send_noteon_many(self.channels, self.notes, self.velocities)
        print("message sent", self.notes)

        return count
