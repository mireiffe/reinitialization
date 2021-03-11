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
[Y, X, Z] = np.indices((128, 96, 15))

imgs = np.where((X - 36) ** 2 + (Y - 53) ** 2 / 2 + (Z - 7) ** 2 * 10 < 400, -1, 1)
m, n, c = imgs.shape

# make image a SDF
ri = Reinitial(dt=.1, width=3, tol=.01, iter=None, dim=3)
phis = ri.getSDF(imgs)

# -----------------------------------------
# visualization
# -----------------------------------------
fig = plt.figure()
mng = plt.get_current_fig_manager()
mng.window.showMaximized()

gr = GridSpec(nrows=3, ncols=5, figure=fig)
ax = []
for i in range(3):
    for j in range(5):
        ax.append(fig.add_subplot(gr[i, j]))
        
        _idx = i * 5 + j
        
        # original image and zero level sets
        ax[_idx].imshow(imgs[..., _idx], 'gray', vmin=0, vmax=1)
        ax[_idx].contour(phis[..., _idx], levels=[0], colors='red')
plt.suptitle('Original image and zero level set of the SDF')

plt.show()

fig.savefig(f"{time.strftime('%H%M-%d%b-%Y', time.localtime(time.time()))}_3D.png")
