import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
from pylab import *
from scipy.io import loadmat


from pyrt_wrapper.midi_listener import MidiListener
import time


class LRTracker:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.stop_run = False


learning_rate = 0.5
tracker = LRTracker(learning_rate=learning_rate)
llrCallback = lambda velocity: set_lrCallback(velocity, tracker)

# Example functions
def set_lrCallback(velocity, tracker):
    tracker.learning_rate = 0.5 * velocity / 127.0
    print(tracker.learning_rate)


def handleNoteInput(isOn, velocity=None):
    if isOn:
        tracker.stop_run = True
        print("noteIsOn:", isOn, velocity)
    else:
        print("noteIsOn:", isOn)


# Example model
actionConfig = {
    48: {"isController": True, "callback": llrCallback},
    # 52: {"isController": True, "callback": handleKnobInput},
    "D3": {"isController": False, "callback": handleNoteInput},
}


images_array = loadmat("IMAGES.mat")["IMAGES"]

A = np.random.rand(256, 256)
V = np.diag(np.linalg.norm(A, axis=1) ** -1)
A = A @ V

K = 50
sz = 16
tracker.learning_rate = 1e-3
lam = 0.2

mixer = np.zeros((256, 256))
for i in range(256):
    x = np.remainder((i - 1), 16) + 1
    y = np.floor((i / 16) + 1)
    for j in range(256):
        x2 = np.remainder(j, 16)
        y2 = np.floor((j) / 16)
        if (np.abs(x - x2) < 5 or np.abs(x - x2) > 13) and (
            np.abs(y - y2) < 5 or np.abs(y - y2) > 13
        ):
            mixer[i, j] = 1

mixer = mixer > 0.5


def grad(A, X, lam, Cp):
    tau = 1
    deltat = 0.05
    time_constants = 5
    eps = 0.5

    a = np.ones((A.shape[1], X.shape[1]))
    last_a = np.zeros((A.shape[1], X.shape[1]))
    G = A.T @ A - np.eye(A.shape[1])

    N = int(tau / deltat * time_constants)

    for ts in range(N):
        a = (1 - deltat / tau) * a + (deltat / tau) * (A.T @ X - G @ a - lam * Cp(a))
        last_a = a
        if np.linalg.norm(a - last_a) < eps * norm(a):
            #             print('good')
            continue

    #     print(np.linalg.norm(a-last_a)/np.linalg.norm(a))

    return a


def extract_image_patches(images_array, sz, K):
    patches = zeros((sz ** 2, K))
    for i in range(K):
        n = np.random.randint(1, 10)
        rx = np.random.randint(0, 511 - sz)
        ry = np.random.randint(0, 511 - sz)
        rand_patch = images_array[rx : rx + sz, ry : ry + sz, n]
        patches[:, i] = np.ndarray.flatten(rand_patch)
    return patches


def Cp(S):
    return np.sign(S)


def Cp2(S):
    return np.multiply(2 * S, (1 + (S ** 2) ** -1))


def CpISA(S):
    F = np.zeros_like(S)
    for i in range(S.shape[0] / 4):
        factor = 0.5 * (np.sum(S[4 * (i + 1) - 4 : 4 * (i + 1) - 1, :] ** 2) ** -0.5)
        F[4 * (i + 1) - 4 : 4 * (i + 1) - 1, :] = np.tile(factor, [4, 1])
    return np.multiply(2 * S, F)


# def Z = CpTICA(S,mixer)
#     F = np.zeros_like(S);
#     for i = 1:size(S,1)
#         factor = 0.5*(sum(S(mixer(i,:),:).^2).^-0.5);
#         F(i,:) = factor;
#     Z = 2*S .* F;


def showbfs(A):
    reA = np.ones((271, 271))
    for i in range(256):
        x = int(np.remainder(i, 16))
        y = int(np.floor(i / 16))
        basis = reshape(A[:, i], [16, 16])
        reA[17 * x : 17 * x + 16, 17 * y : 17 * y + 16] = basis / (
            np.max(np.abs(np.ndarray.flatten(basis)))
        )

    return reA


A = np.random.rand(256, 256)
V = np.diag(np.linalg.norm(A, axis=1) ** -1)
A = A @ V


def make_montage_2(Images):
    n_el = Images.shape[0]
    hdim = int(np.sqrt(n_el))
    montage = np.hstack(
        np.split(np.vstack(Images[:n_el].reshape(-1, 16, 16)), 16, axis=0)
    )
    im = cv2.resize(montage, (400, 400))
    cv2.imshow("frame", im)


def make_montage(Images, epoch, lr):
    n_small_im = Images.shape[1]
    n_pixels_side = int(np.sqrt(Images.shape[0]))
    n_small_im_sqrt = int(np.sqrt(n_small_im))
    montage = np.ones(
        [
            n_small_im_sqrt * n_pixels_side + (n_small_im_sqrt - 1),
            n_small_im_sqrt * n_pixels_side + (n_small_im_sqrt - 1),
        ]
    )
    for image in range(n_small_im):
        small_im = Images[:, image]
        small_im = small_im.reshape(n_pixels_side, n_pixels_side)
        small_im = small_im / (np.abs(small_im)).max()
        pos1 = int(image // n_small_im_sqrt)
        pos2 = int(image % n_small_im_sqrt)
        montage[
            pos1 * (n_pixels_side + 1) : pos1 * (n_pixels_side + 1) + n_pixels_side,
            pos2 * (n_pixels_side + 1) : pos2 * (n_pixels_side + 1) + n_pixels_side,
        ] = small_im
    # if epoch == 1:
    #     plt.figure(figsize=(15, 15))
    # plt.ion()
    # plt.imshow(montage, cmap="gray", vmin=-1, vmax=1)
    # plt.title("Epoch: {}, LR: {}".format(epoch, lr))
    # plt.draw()
    # plt.pause(0.001)
    # print(montage.shape)
    im = cv2.resize(montage, (400, 400))
    cv2.imshow("frame", im)


A = np.random.rand(256, 256) - 0.5
V = np.diag(np.linalg.norm(A, axis=0) ** -1)
A = A @ V


#### listen
# Pass the action config and an optional second boolean for verbose logging
listener = MidiListener(actionConfig, True)
listener.start()


i = 0
while not tracker.stop_run:
    X = extract_image_patches(images_array, sz, K)
    S = grad(A, X, lam, Cp)
    #     print(S)
    A = (1 - tracker.learning_rate) * A + tracker.learning_rate * (X - (A @ S)) @ S.T
    #     print(i,np.mean(np.linalg.norm(A,axis=0)))

    V = np.diag(1.0 / np.linalg.norm(A, axis=0))
    A = A @ V

    i = i + 1

    if i % 1 == 0:
        # print("n_epochs=" + str(i))
        make_montage(A, epoch=i, lr=tracker.learning_rate)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

listener.stop()
cv2.destroyAllWindows()

#
# # create two subplots
# ax1 = plt.subplot(1, 2, 1)
# ax2 = plt.subplot(1, 2, 2)
#
#
# # create two image plots
# im1 = ax1.imshow(grab_frame(cap1))
# im2 = ax2.imshow(grab_frame(cap2))
#
#
# def update(i):
#     im1.set_data(grab_frame(cap1))
#     im2.set_data(grab_frame(cap2))
#
#
# ani = FuncAnimation(plt.gcf(), update, interval=200)
# plt.show()
