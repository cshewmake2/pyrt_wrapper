from pyrt_wrapper.midi_listener import MidiListener
import time


class Tracker:
    def __init__(self, s=0.0):
        self.s = s


s = 0.0
tracker = Tracker(s=s)
llrCallback = lambda velocity: set_lrCallback(velocity, tracker)

# Example functions
def set_lrCallback(velocity, tracker):
    tracker.s = 2 * 3.14159 * (2 * (velocity / 127.0) - 1)
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


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

dt = 0.005
n = 20
L = 1
particles = np.zeros(
    n,
    dtype=[
        ("position", float, 2),
        ("velocity", float, 2),
        ("force", float, 2),
        ("size", float, 1),
    ],
)

particles["position"] = np.random.uniform(0, L, (n, 2))
particles["velocity"] = np.zeros((n, 2))
particles["size"] = 0.5 * np.ones(n)

fig = plt.figure(figsize=(7, 7))
ax = plt.axes(xlim=(0, L), ylim=(0, L))
scatter = ax.scatter(particles["position"][:, 0], particles["position"][:, 1])


def update(frame_number):
    particles["force"] = tracker.s * np.random.uniform(0, 10, (n, 2))
    particles["velocity"] = particles["velocity"] + particles["force"] * dt
    particles["position"] = particles["position"] + particles["velocity"] * dt

    particles["position"] = particles["position"] % L
    scatter.set_offsets(particles["position"])
    return (scatter,)


anim = FuncAnimation(fig, update, interval=10)
plt.show()
#

#
# # Learning Code
# x0 = 100
# for i in range(0, 1000):
#     x0 = x0 - tracker.learning_rate * df(x0)
#     time.sleep(0.1)
#     print("x: {}, f(x): {}, lr: {}".format(x0, ff(x0), tracker.learning_rate))
#
# # print("Hello")
#
#
# print("Printed immediately.")

listener.stop()
