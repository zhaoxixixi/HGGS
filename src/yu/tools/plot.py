"""
@author: longlong.yu
@email: yulonglong.hz@qq.com
@date: 2023-03-29
@description: convenient tools for save or show plot which is constructed with matplotlib
"""
import os
from typing import List

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt, colormaps
from sklearn.manifold import TSNE

from yu.const.normal import DisplayType
from yu.tools.misc import makedir
from yu.const.normal import ConstData


def _safe_obtain_array(array, *args):
    """  """
    current = array
    for index in args:
        if current is not None and isinstance(current, (np.ndarray, list)) and index < len(current):
            current = current[index]
        else:
            return None
    return current


def _construct_plot(
        ys, xs,
        labels=None,
        x_labels=None,
        y_labels=None,
        titles=None,
        callbacks=None,
        colors=None
):
    """ construct plot """
    amount = len(ys)
    # amount = 1
    fig, axs = plt.subplots(amount, 1)
    # fig, axs = plt.subplots(1, 1)
    if amount == 1:
        axs = [axs]

    # for i in range(1):
    for i in range(amount):
        title = _safe_obtain_array(titles, i)
        if title:
            axs[i].set_title(titles[i])

        curve_amount = min(len(ys[i]), len(xs[i]))
        for j in range(curve_amount):
            label = _safe_obtain_array(labels, i, j)
            color = _safe_obtain_array(colors, j)
            tmp = {}
            if color:
                tmp['color'] = color
            if label:
                tmp['label'] = label
            axs[i].plot(xs[i][j], ys[i][j], lw=4.0, **tmp)  # , lw=4.0

        x_label = _safe_obtain_array(x_labels, i)
        if x_label:
            axs[i].set_xlabel(x_label)
        y_label = _safe_obtain_array(y_labels, i)
        if y_label:
            axs[i].set_ylabel(y_label)
        callback = _safe_obtain_array(callbacks, i)
        if callback:
            callback(fig, axs[i], xs[i], ys[i])
        if labels is not None:
            axs[i].legend(loc='upper right')

    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.5)
    return fig, axs


def show_plot(
        ys, xs,
        labels=None,
        x_labels=None,
        y_labels=None,
        titles=None,
        callbacks=None,
        colors=None
):
    """ plot img """
    fig, _ = _construct_plot(ys, xs, labels, x_labels, y_labels, titles, callbacks=callbacks, colors=colors)
    plt.show()
    plt.close(fig)


def save_plot(
        ys, xs, save_name,
        labels=None,
        x_labels=None,
        y_labels=None,
        titles=None,
        callbacks=None,
        colors=None,
):
    """ save img """
    makedir(os.path.dirname(save_name))
    fig, _ = _construct_plot(ys, xs, labels, x_labels, y_labels, titles, callbacks=callbacks, colors=colors)
    for ax in _:  # erase the legend
        ax.axis('off')
    fig.savefig(save_name,  bbox_inches='tight', dpi=2048)
    plt.close(fig)


def draw_random_x_marks(ax, num_x=10, x_size=0.5):
    import random
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    
    for _ in range(num_x):
        x_center = random.uniform(x_lim[0], x_lim[1])
        y_center = random.uniform(y_lim[0], y_lim[1])
        
        x1, x2 = x_center - x_size / 2, x_center + x_size / 2
        y1, y2 = y_center - x_size / 2, y_center + x_size / 2
        
        ax.plot([x1, x2], [y1, y2], color='black')
        ax.plot([x1, x2], [y2, y1], color='black')

def draw_scatter(
        xs, ys, c,
        x_name: str, y_name: str,
        xlim, ylim,
        title: str, save_name: str,
        display: DisplayType,
        vmin: float=0.0,
        vmax: float=1.0,
        random_Fork: bool=False
):
    """ 2d scatter """
    # xs = torch.tensor(xs)
    # ys = torch.tensor(ys)
    cmap = colormaps['coolwarm']
    if 'MAE' in title:
        # print(colormaps)
        cmap = colormaps['gist_gray_r']
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    if c:
        # c = torch.tensor(c)
        scatter = plt.scatter(xs, ys, c=c, s=1, cmap=cmap, norm=norm)
    else:
        scatter = plt.scatter(xs, ys, s=1, cmap=cmap, norm=norm)
    plt.colorbar(scatter)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.title(title)

    if random_Fork:
        ax = plt.gca()
        draw_random_x_marks(ax, 150, 0.025)

    if display == DisplayType.SHOW:
        plt.show()
    elif display == DisplayType.SAVE:
        plt.savefig(save_name)
    plt.close()



