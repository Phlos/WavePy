# plotting shizzle (keep it neat)

from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import copy

from utility_functions import find_nearest

from WavePyClasses import Grid, Source, Receiver

def plot_field(
    grid: Grid, 
    field, 
    sources: Union[Source, List[Source], None] = None, 
    receivers: Union[Receiver, List[Receiver], None] = None,
    title=None, colorbar=False, 
    cmaks=None, 
    ax=None, draw=True, updating=False,
    **kwargs
):
    
    '''
    Plot the velocity field in a somewhat nice way
    '''
    
    if not ax:
        fig, ax = plt.subplots(1, figsize=(10,5))
    else:
        plt.cla()

    if not title:
        print('no plot title given')

    # plot any field
    X = grid.X / 1000.
    Z = grid.Z / 1000.

    if not ax:
        fig, ax = plt.subplots(1, figsize=(10,5))

    if cmaks:
        img = ax.pcolormesh(X,Z, field, shading='gouraud',
                            vmin=-cmaks, vmax=cmaks, **kwargs)
    else:
        img = ax.pcolormesh(X,Z, field, shading='gouraud', **kwargs)

    for src in sources:
        loc = src.location.asarray() /1000.
        ax.plot(*loc, c= 'k', marker='*')
    for rec in receivers:
        loc = rec.location.asarray() /1000.
        ax.plot(*loc, c='k', marker='^')

    ax.set_xlabel('horizontal distance [km]')
    ax.set_ylabel('depth [km]')
    if title:
        ax.set_title(title)
    if not updating:  # this is a really ugly hack to make sure that the axis isn't continuously swapping up & down, but it depends entirely on the version of python, matplotlib, and whatnot.
        ax.invert_yaxis()

    ax.axis('image')

    if colorbar:
      plt.colorbar(mappable=img, ax=ax)

    fig = plt.gcf()
    if draw:
        fig.canvas.draw()

    return img, ax, fig
