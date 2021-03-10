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

# make a x b patches for the img
a, b = 2, 3
lst_imgs = [img[m // a * i:m // a * (i + 1), n // b * j:n // b * (j + 1)]
        for i in range(a) for j in range(b)]
imgs = np.stack(lst_imgs, axis=-1)

# make image a SDF
ri = Reinitial(imgs, dt=.1, width=10, tol=.001, iter=None, dim=2)
phis = ri.getSDF()

# to check that norm of gradient is 1
phis_x = .5 * cv2.Sobel(phis, -1, 1, 0, ksize=1, borderType=cv2.BORDER_REFLECT)
phis_y = .5 * cv2.Sobel(phis, -1, 0, 1, ksize=1, borderType=cv2.BORDER_REFLECT)
ng = np.sqrt(phis_x ** 2 + phis_y ** 2)

# -----------------------------------------
# visualization
# -----------------------------------------
phis = phis.reshape((m // a, n // b, a, b))
ng = ng.reshape((m // a, n // b, a, b))

fig = plt.figure()
mng = plt.get_current_fig_manager()
mng.window.showMaximized()

ax = fig.add_subplot(111)

# original image
ax.imshow(img, 'gray')

# zero level sets on the original locations
cmaps = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
for i in range(a):
    for j in range(b):
        _ps = np.zeros_like(img)
        _ps[m // a * i:m // a * (i + 1), n // b * j:n // b * (j + 1)] = phis[..., i, j]
        ax.contour(_ps, levels=[0], colors=cmaps[i * b + j])
ax.set_title('Original image and zero level set of the SDF')

plt.show()

# fig.savefig(f"{time.strftime('%H%M-%d%b-%Y', time.localtime(time.time()))}_2D.png")
