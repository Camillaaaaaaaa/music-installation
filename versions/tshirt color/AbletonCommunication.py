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

        if message:
            # return (messagetype+channel, value1, value2)
            return message[0]
        else:
            return False

    def send_message(self, message_type, channel, value1, value2):
        if message_type == "CC":
            message = ([CONTROL_CHANGE | channel, value1, value2])
            self.midiout.send_message(message)


    def setModulation(self):
        value=0
        while True:
            mod = ([CONTROL_CHANGE | 8, 76, value])
            self.midiout.send_message(mod)
            #print(value)
            #value-=1
            time.sleep(3)

#mod=AbletonCommunication()
#mod.setModulation()