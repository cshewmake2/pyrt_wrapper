from midi_listener import MidiListener
import time


class LRTracker:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate


learning_rate = 0.1
tracker = LRTracker(learning_rate=learning_rate)
llrCallback = lambda velocity: set_lrCallback(velocity, tracker)

# Example functions
def set_lrCallback(velocity, tracker):
    tracker.learning_rate = 5 * velocity / 127.0
    # print(tracker.learning_rate)


# def handleNoteInput(isOn, velocity=None):
#     if velocity:
#         return print("noteIsOn:", isOn, velocity)
#     else:
#         return print("noteIsOn:", isOn)


# Example model
actionConfig = {
    48: {"isController": True, "callback": llrCallback},
    # 52: {"isController": True, "callback": handleKnobInput},
    # "D3": {"isController": False, "callback": handleNoteInput},
}

# Pass the action config and an optional second boolean for verbose logging
listener = MidiListener(actionConfig, True)
listener.start()


### Define Problem
ff = lambda x: (x - 4) ** 2
df = lambda x: 2 * (x - 4)


# Learning Code
x0 = 100
for i in range(0, 1000):
    x0 = x0 - tracker.learning_rate * df(x0)
    time.sleep(0.1)
    print("x: {}, f(x): {}, lr: {}".format(x0, ff(x0), tracker.learning_rate))

# print("Hello")
#
#
# print("Printed immediately.")

listener.stop()
