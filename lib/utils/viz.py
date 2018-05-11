from __future__ import absolute_import, division

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def show_frame(image, bndbox=None, fig_n=1, pause=0.001,
               thickness=5, cmap=None):
    global fig_dict
    if not 'fig_dict' in globals():
        fig_dict = {}

    if bndbox is not None:
        image = image.copy()
        draw = ImageDraw.Draw(image)
        color = (255, 0, 0) if image.mode == 'RGB' else 255
        for t in range(-thickness // 2, thickness // 2 + 1):
            draw.rectangle((
                int(bndbox[0] + t),
                int(bndbox[1] + t),
                int(bndbox[0] + bndbox[2] + t),
                int(bndbox[1] + bndbox[3] + t)),
                outline=color)

    if not fig_n in fig_dict:
        fig = plt.figure(fig_n)
        plt.axis('off')
        fig.tight_layout()
        fig_dict[fig_n] = plt.imshow(image, cmap=cmap)
    else:
        fig_dict[fig_n].set_data(image)

    plt.pause(pause)
    plt.draw()

    return image
