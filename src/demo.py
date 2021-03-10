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

from reinitial import reinitial
# load an image from url, that contains multiple polygons
img_pil = Image.open(requests.get("https://i.stack.imgur.com/7Qnug.jpg", stream=True).raw)
img = np.where(np.array(img_pil)[..., 0] < 100, -1, 1)
m, n = img.shape

# make image a SDF
ri = reinitial(img, dt=.1, width=3, tol=.001, iter=None, dim=2)
phi = ri.getSDF()

# to check that norm of gradient is 1
phi_x = .5 * cv2.Sobel(phi, -1, 1, 0, ksize=1, borderType=cv2.BORDER_REFLECT)
phi_y = .5 * cv2.Sobel(phi, -1, 0, 1, ksize=1, borderType=cv2.BORDER_REFLECT)
ng = np.sqrt(phi_x ** 2 + phi_y ** 2)

# visualization
fig = plt.figure()
mng = plt.get_current_fig_manager()
mng.window.showMaximized()

gr = GridSpec(nrows=2, ncols=3, figure=fig)
ax = []
ax.append(fig.add_subplot(gr[0, 0]))    # image and zero level set
ax.append(fig.add_subplot(gr[1, 0]))    # norm of gradient
ax.append(fig.add_subplot(gr[:, 1:], projection='3d'))   # surface

# image and zero level set
ax[0].imshow(img, 'gray')
ax[0].contour(phi, levels=0, colors='red')
ax[0].set_title('Original image and zero level set of the SDF')

# norm of gradient
nga = ax[1].imshow(ng)
fig.colorbar(nga, ax=ax[1], shrink=.7, aspect=10)
ax[1].set_title('Norm of gradient')

# 3D surface
y, x = np.indices((m, n))
surf = ax[2].plot_surface(x, y, phi, rcount=100, ccount=100, cmap=cm.jet)
# surf = ax[2].plot_wireframe(x, y, phi, rcount=100, ccount=100, cmap=cm.jet)
ax[2].zaxis.set_major_locator(LinearLocator(10))
fig.colorbar(surf, ax=ax[2], shrink=.75, aspect=10)
ax[2].set_title('3D surface of the SDF')

# fig.savefig(f"{time.strftime('%H%M-%d%b-%Y', time.localtime(time.time()))}.png")

plt.show()

