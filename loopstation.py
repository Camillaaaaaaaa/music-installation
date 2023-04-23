import time
import pygame
import os


class LoopStation:
    def __init__(self):
        self.bpm = 90
        self.len_beat = 60 / self.bpm
        self.len_music = 120

        # array for each beat in one 8 count with what tone(s) to play
        self.tones = [[], [], [], [], [], [], [], []]

        pygame.init()
        self.beep = pygame.mixer.Sound("sounds/beep.wav")
        self.bop = pygame.mixer.Sound("sounds/bop.wav")
        pygame.mixer.Sound.set_volume(self.beep, 0.2)
        pygame.mixer.Sound.set_volume(self.bop, 0.2)

        self.allTones = {}
        list_of_files = []
        for root, dirs, files in os.walk("sounds/tones/"):
            for file in files:
                list_of_files.append(os.path.join(root, file))
        for file in list_of_files:
            self.allTones[file[13:15]] = pygame.mixer.Sound(file)

        print(self.len_beat)

    def set_tones(self, new_tones):
        self.tones = new_tones

    def play_tone(self, count):
        pygame.mixer.stop()
        if self.tones[count - 1]:
            for tone in self.tones[count - 1]:
                pass
                self.allTones[tone].play()

        if count == 1:
            self.beep.play()
            count += 1
        else:
            self.bop.play()
            count += 1
            if count > 8:
                count = 1
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
