import time
import pygame
import os
import rtmidi2


class LoopStation:
    def __init__(self, amount_beats, bpm):
        self.bpm = bpm
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
        self.midiout.open_port(3)

        # instrument mappings

        # bass drum: C1, rim shot: C#1, Snare Drum: D1, Hand Clap: D#1, Snare Drum: G1, Ride:D2
        drums = {5: [36], 4: [37], 3: [38], 2: [39], 1: [43], 0: [50]}

        # D3, F3, A3, C4, E4, G4
        piano = {5: [62], 4: [65], 3: [69], 2: [72], 1: [76], 0: [79]}

        # D0, F0, A0, C1, E1, G1
        bass = {5: [26], 4: [29], 3: [33], 2: [36], 1: [40], 0: [43]}

        #  D2, F2, A2, C3, E3, G3
        guitar = {5: [50], 4: [53], 3: [57], 2: [60], 1: [64], 0: [67]}

        # [D2,F2,A2], [F2,A2,C3], [A2,C3,E3], [C3,E3,G3], [E3,G3,Bb3], [G3,Bb3,D4]
        piano_chords = {5: [50, 53, 57], 4: [53, 57, 60], 3: [57, 60, 64], 2: [60, 64, 70], 1: [64, 70, 73],
                        0: [79, 82, 86]}
        # [D3,F3,A3], [F3,A3,C4], [A3,C4,E4], [C4,E4,G4], [E4,G4,Bb4], [G4,Bb4,D5]
        # piano_chords = {5: [62, 65, 69], 4: [65, 69, 73], 3: [69, 72, 76], 2: [72, 76, 79], 1: [76, 79, 82],0: [79, 82, 86]}

        guitar_chords = {5: [50, 53, 57], 4: [53, 57, 60], 3: [57, 60, 64], 2: [60, 64, 70], 1: [64, 70, 73],
                         0: [79, 82, 86]}

        # 1: Hight hat:F#1, bass drum: C1, Snare Drum D1
        base_drums = {1: [42], 2: [36], 3: [38]}

        # D3, F3, A3, C4, E4, G4
        xylophone = {5: [62], 4: [65], 3: [69], 2: [72], 1: [76], 0: [79]}

        self.instruments = [drums, piano, bass, guitar, xylophone, piano_chords, guitar_chords, base_drums]

    def play_half_beat(self, count):
        for note in self.tones[0][count - 1]:
            if 36 in self.instruments[0][note] or 37 in self.instruments[0][note]:
                self.midiout.send_noteon(0, 36, 100)

        for note in self.tones[7][count - 1]:
            self.midiout.send_noteon(0, 42, 100)

    def set_bpm(self, bpm):
        self.bpm = bpm
        self.len_beat = 60 / self.bpm

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
                for n in self.instruments[i][note]:
                    if i == 5:
                        self.channels.append(1)
                        self.velocities.append(80)
                    elif i == 6:
                        self.channels.append(3)
                        self.velocities.append(80)
                    elif i == 7:
                        self.channels.append(0)
                        self.velocities.append(100)
                    else:
                        self.channels.append(i)
                        self.velocities.append(100)
                    self.notes.append(n)

        self.midiout.send_noteon_many(self.channels, self.notes, self.velocities)
        print("message sent", self.notes)
        print("channels", self.channels)

        return count
