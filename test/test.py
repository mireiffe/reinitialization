import time
import requests

import cv2
import numpy as np
from PIL import Image

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LinearLocator

from reinitial import Reinitial


# load an image from url, that contains multiple polygons
img_pil = Image.open(requests.get("https://i.stack.imgur.com/7Qnug.jpg", stream=True).raw)
img = np.where(np.array(img_pil)[..., 0] < 100, -1, 1)
m, n = img.shape

# make image a SDF
ri = Reinitial(img, dt=.1, width=3, tol=.01, iter=None, dim=2, debug=True)
phi = ri.getSDF()

# to check that norm of gradient is 1
phi_x = .5 * cv2.Sobel(phi, -1, 1, 0, ksize=1, borderType=cv2.BORDER_REFLECT)
phi_y = .5 * cv2.Sobel(phi, -1, 0, 1, ksize=1, borderType=cv2.BORDER_REFLECT)
ng = np.sqrt(phi_x ** 2 + phi_y ** 2)

# -----------------------------------------
# visualization
# -----------------------------------------
fig = plt.figure()
mng = plt.get_current_fig_manager()
mng.window.showMaximized()

gr = GridSpec(nrows=2, ncols=2, figure=fig)
ax = []
ax.append(fig.add_subplot(gr[0, 0]))    # image and zero level set
ax.append(fig.add_subplot(gr[1, 0]))    # SDF
ax.append(fig.add_subplot(gr[1, 1]))    # norm of gradient

# image and zero level set
ax[0].imshow(img, 'gray')
ax[0].contour(phi, levels=[0], colors='red')
ax[0].set_title('Original image and zero level set of the SDF')

# SDF
sdf = ax[1].imshow(phi)
fig.colorbar(sdf, ax=ax[1], shrink=.7, aspect=10)
ax[1].set_title('SDF')

# norm of gradient
nga = ax[2].imshow(ng)
fig.colorbar(nga, ax=ax[2], shrink=.7, aspect=10)
ax[2].set_title('Norm of gradient')

pass
plt.show()