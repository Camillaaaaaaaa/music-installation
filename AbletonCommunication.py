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


    def get_message(self):
        # returns midi message from ableton
        message = self.midiin.get_message()
        # [messagetype+channel, value1, value2]
        # 176=CC
        # 144= note on
        # 128= note off

        if message:
            # checks if note is turned on or off on tempo track meaning new beat at started
            if message[0][0] == 144 + 15:
                # calculate count of beat from note height
                value = int((message[0][1] - 41) * 2)
                return [True, value]
            if message[0][0] == 128 + 15:
                value = int((message[0][1] - 41) * 2 + 1)
                return [True, value]

    def send_message(self, message_type, channel, value1, value2):
        if message_type == "CC":
            message = ([CONTROL_CHANGE | channel, value1, value2])
            self.midiout.send_message(message)

    def setModulation(self, channel):
        # create MIDI mapping in ableton to specific channel
        # turn on MIDI learn on in ableton
        # select what should be mapped (volume/pitch...)
        # run function to map it to specific channel to be able to control it later with MIDI messages
        value = 0
        while True:
            mod = ([CONTROL_CHANGE | channel, 76, value])
            self.midiout.send_message(mod)
            time.sleep(0.1)

# mod=AbletonCommunication()
# mod.setModulation(8)
