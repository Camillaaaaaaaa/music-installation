import rtmidi
from rtmidi.midiconstants import (CONTROL_CHANGE)
import time


class AbletonCommunication():
    def __init__(self):
        self.port_P_to_A = 2
        self.port_A_to_P = 0

        self.midiout = rtmidi.MidiOut()
        ports = self.midiout.get_ports()
        print("midiout ports:", ports)
        self.midiout.open_port(self.port_P_to_A)

        self.midiin = rtmidi.MidiIn()
        ports = self.midiin.get_ports()
        print("midiin ports:", ports)
        self.midiin.open_port(self.port_A_to_P)

    def reopen_midi_in(self):
        self.midiin.close_port()
        self.midiin.open_port(self.port_A_to_P)
        self.midiout.close_port()
        self.midiout.open_port(self.port_P_to_A)

    def get_message(self):
        # returns midi message from ableton
        message = self.midiin.get_message()

        if message:
            if message[0][0] == 144 + 15:
                value = int((message[0][1] - 41) * 2)
                return [True, value]
            if message[0][0] == 128 + 15:
                value = int((message[0][1] - 41) * 2 + 1)
                return [True, value]

    def send_message(self, message_type, channel, value1, value2):
        if message_type == "CC":
            message = ([CONTROL_CHANGE | channel, value1, value2])
            self.midiout.send_message(message)

    def setModulation(self):
        value = 0
        while True:
            mod = ([CONTROL_CHANGE | 8, 76, value])
            self.midiout.send_message(mod)
            # print(value)
            # value-=1
            time.sleep(3)

# mod=AbletonCommunication()
# mod.setModulation()
