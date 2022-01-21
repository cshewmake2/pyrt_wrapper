import numpy as np
import matplotlib.pyplot as plt

from random import choices

import torch.nn as nn
import torch
import time
from geom_tools.geometry import MatrixLieGroup

from midi_listener import MidiListener
from matplotlib.animation import FuncAnimation


class Tracker:
    def __init__(
        self,
        s1=torch.zeros(1).unsqueeze(0),
        s2=torch.zeros(1).unsqueeze(0),
        s3=torch.zeros(1).unsqueeze(0),
        s4=torch.zeros(1).unsqueeze(0),
        s5=torch.zeros(1).unsqueeze(0),
    ):
        self.s = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}
        self.stop_run = False


tracker = Tracker()
llrCallback1 = lambda velocity: set_lrCallback(velocity, tracker, "s1")
llrCallback2 = lambda velocity: set_lrCallback(velocity, tracker, "s2")
llrCallback3 = lambda velocity: set_lrCallback(velocity, tracker, "s3")
llrCallback4 = lambda velocity: set_lrCallback(velocity, tracker, "s4")
llrCallback5 = lambda velocity: set_lrCallback(velocity, tracker, "s5")

# Example functions
def set_lrCallback(velocity, tracker, key):
    tracker.s[key] = torch.tensor([2 * (velocity / 127.0) - 1]).unsqueeze(0)
    print(tracker.s[key])
    # print(tracker.learning_rate)


def handleNoteInput(isOn, velocity=None):
    if isOn:
        tracker.stop_run = True
        print("noteIsOn:", isOn, velocity)
    else:
        print("noteIsOn:", isOn)


# Example model
actionConfig = {
    48: {"isController": True, "callback": llrCallback1},
    49: {"isController": True, "callback": llrCallback2},
    50: {"isController": True, "callback": llrCallback3},
    51: {"isController": True, "callback": llrCallback4},
    52: {"isController": True, "callback": llrCallback5},
    # 52: {"isController": True, "callback": handleKnobInput},
    "D3": {"isController": False, "callback": handleNoteInput},
}

# Pass the action config and an optional second boolean for verbose logging
listener = MidiListener(actionConfig, True)
listener.start()


def generate_gabor_values(g_grid, gamma=1, sigma=0.5, psi=0, lmbda=1):

    yscaled_g_grid = g_grid * torch.Tensor([1, gamma ** 2])
    gaussian_arg = (g_grid * yscaled_g_grid).sum(-1) / (-2 * sigma ** 2)
    gaussian_term = torch.exp(gaussian_arg)

    sinusoid_arg = 1j * (
        (2 * np.pi / lmbda) * (g_grid * torch.Tensor([1, 0]))[:, :, :, 0] + psi
    )
    sinusoid_term = torch.exp(sinusoid_arg)

    gabor = gaussian_term * sinusoid_term
    return gabor


def generate_grid(xyrange=1, scale=0.1, n_samples=5):
    x = scale * torch.linspace(-xyrange, xyrange, n_samples)
    y = scale * torch.linspace(-xyrange, xyrange, n_samples)
    xx, yy = torch.meshgrid(x, y)
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).float()
    return grid


g_grid = generate_grid(1, 1, 512)

lie_algebra = torch.Tensor([[[0, -1], [1, 0]]])
grp = MatrixLieGroup(lie_algebra, device="cpu")
rg_grid, g = grp.action(g_grid.reshape(-1, 2), tracker.s["s1"])

# tracker.s = torch.Tensor([np.pi / 6]).unsqueeze(0)


### PLOT GRID POINTS
# fig = plt.figure(figsize=(7, 7))
# lims = 2
# ax = plt.axes(xlim=(-lims, lims), ylim=(-lims, lims))
# scatter = ax.scatter(rg_grid[:, 0], rg_grid[:, 1])
# def update(frame_number):
# rg_grid, g = grp.action(g_grid.reshape(-1, 2), tracker.s)
# scatter.set_offsets(rg_grid)
# return (scatter,)

### PLOT GABOR IMAGE
gabor = generate_gabor_values(
    g_grid, gamma=1, sigma=0.5
)  # tracker.s["s1"], sigma=tracker.s["s2"])
fig = plt.figure(figsize=(7, 7))
im = plt.imshow(gabor.real.squeeze(0), animated=True)


def update(*args):
    if tracker.stop_run:
        listener.stop()
        raise Error("Simulation Stop Occurred")
    gabor = generate_gabor_values(
        g_grid,
        gamma=tracker.s["s1"],
        sigma=tracker.s["s2"],
        psi=tracker.s["s4"],
        lmbda=tracker.s["s5"],
    )
    rot = torch.exp(1j * tracker.s["s3"])
    rot_gabor = rot * gabor
    im.set_array(rot_gabor.real.squeeze(0).numpy())
    return (im,)


anim = FuncAnimation(fig, update, interval=25, blit=True)
plt.show()


# listener.stop()

#
# plt.imshow(gabor.real[0], norm=None)
# plt.show()
#
# plt.imshow(gabor.imag[0], norm=None)
# plt.show()
#
# gabor = generate_gabor_values(g_grid, gamma=0.4, sigma=0.2)
#
# # plt.imshow(gabor.real[0], norm=None)
# # plt.show()
# #
# # plt.imshow(gabor.imag[0], norm=None)
# # plt.show()
#
# import svis
#
# frames = []
# for s in np.linspace(0, np.pi * 2):
#     rot = np.exp(1j * s)
#     rot_gabor = rot * gabor
#
#
#     frames.append(rot_gabor.real)


# frames = torch.cat(frames, dim=0)
# svis.animated_video(frames.numpy())

# iterparam = torch.linspace(0,np.pi,64)
# gabors = []
# for ip in iterparam:
#     rg_grid, g = grp.action(g_grid.reshape(-1,2),ip.unsqueeze(0).unsqueeze(0))
#     gabors.append(generate_gabor_values(rg_grid.reshape(1,16,16,2))[0].real)

# gabors = torch.stack(gabors)

######
# A_rot = torch.Tensor([[[0,-1],[1,0]]])
# rot_grp = MatrixLieGroup(lie_algebra=A_rot,device = 'cpu')
#
# A_sc = torch.Tensor([[[1,0],[0,1]]])
# sc_grp = MatrixLieGroup(lie_algebra=A_sc,device='cpu')
#
# # gamma_vals = torch.linspace(0.1,2.1,68)
# thetas = torch.linspace(0,np.pi,64)
# scales = torch.linspace(0,0,1)
# gabors = torch.Tensor([])
# first = True
# for th in thetas:
#     for sc in scales:
#         rg_grid, g = sc_grp.action(g_grid.reshape(-1,2),sc.unsqueeze(0).unsqueeze(0))
#         rg_grid, g = rot_grp.action(rg_grid.reshape(-1,2),th.unsqueeze(0).unsqueeze(0))
#         inner_gabor = generate_gabor_values(rg_grid.reshape(1,16,16,2))[0].real.unsqueeze(0)
#         if first:
#             gabors = inner_gabor
#             first = False
#         else:
#             gabors = torch.cat([gabors,inner_gabor],dim=0)


# gabors = torch.stack(gabors)
# gabors = torch.stack([generate_gabor_values(g_grid, psi=pi)[0].real for pi in iterparam])
# gabors = torch.stack([generate_gabor_values(rg_grid_i.reshape(1,16,16,2), psi=pi)[0].real for rg_grid_i in rg_grid])


# rg_grid, g = sc_grp.action(g_grid.reshape(-1,2),thetas.unsqueeze(-1))
#
# def gen_gabors(theta, scales):
#     rg_grid, g = sc_grp.action(g_grid.reshape(-1,2),sc.unsqueeze(0).unsqueeze(0))
#     rg_grid, g = rot_grp.action(rg_grid.reshape(-1,2),th.unsqueeze(0).unsqueeze(0))
#     generate_gabor_values(rg_grid.reshape(1,16,16,2))[0].real
