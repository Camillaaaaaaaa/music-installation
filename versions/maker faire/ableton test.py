
import rtmidi2

midiout = rtmidi2.MidiOut()
ports = rtmidi2.get_out_ports()
print(ports)
midiout.open_port(1)

midiout.send_noteon(0, 36, 100)