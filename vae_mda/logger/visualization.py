import numpy as np
from matplotlib import pylab as P
from matplotlib import pyplot as plt


def show_vae(
    before: np.array,
    after: np.array,
    title=None,
): 
    figure, axs = plt.subplots(1, 3, figsize=(15, 5))
    _ = [ax.set_facecolor('grey') for ax in axs]
    
    vmin, vmax = np.nanmin(before), np.nanmax(before)
    delta = np.nanmax(np.absolute(after-before))

    show_x(
        before,
        title='Before',
        ax=P.subplot(1, 3, 1),
    )
    show_x(
        after,
        title='After',
        ax=P.subplot(1, 3, 2),
        vmin=vmin,
        vmax=vmax,
    )
    show_x(
        after-before,
        title='Difference',
        ax=P.subplot(1, 3, 3),
        vmin=-delta,
        vmax=delta,
        cmap='RdBu',
    )
    figure.suptitle(title)
    return figure


def show_assimilation(
    background: np.array,
    observation: np.array,
    result: np.array,
    validation: np.array,
    title=None,
):
    figure, axs = plt.subplots(2, 3, figsize=(12, 9))
    _ = [ax.set_facecolor('grey') for ax in axs[0]]
    _ = [ax.set_facecolor('grey') for ax in axs[1]]

    vmin = np.nanmin(background)
    vmax = np.nanmax(background)
    delta = max(
        np.nanmax(np.abs(result-background)),
        np.nanmax(np.abs(result-validation)),
    )

    if observation is not None:
        show_xy(
            background,
            observation,
            title='Background with observation',
            ax=P.subplot(2, 3, 1),
        )
    else:
        show_x(
            background,
            title='Background',
            ax=P.subplot(2, 3, 1),
        )
    show_x(
        result,
        title='Assimilation result',
        ax=P.subplot(2, 3, 2),
        vmin=vmin,
        vmax=vmax,
    )
    show_x(
        result-background,
        title='Result background difference',
        ax=P.subplot(2, 3, 3), 
        cmap='RdBu',
        vmin=-delta,
        vmax=delta,
    )
    show_x(
        validation,
        title='Validation',
        ax=P.subplot(2, 3, 4)
    )
    show_x(
        result,
        title='Assimilation result',
        ax=P.subplot(2, 3, 5),
        vmin=vmin,
        vmax=vmax,
    )
    show_x(
        result-validation,
        title='Result validation difference',
        ax=P.subplot(2, 3, 6),
        cmap='RdBu',
        vmin=-delta,
        vmax=delta,
    )
    figure.suptitle(title)
    return figure


def show_x(
    x: np.array,
    title='',
    ax=None,
    cmap='Blues_r',
    vmin=None,
    vmax=None,
):
    if ax is None:
        P.figure()
    P.imshow(x.T, cmap=cmap, vmin=vmin, vmax=vmax) 
    P.colorbar(orientation='horizontal')
    P.title(title)

def show_xy(
    x: np.array,
    y: np.array,
    title='',
    ax=None,
    cmap='Blues_r',
    vmin=None,
    vmax=None,
):
    if ax is None:
        P.figure()
    P.imshow(x.T, cmap=cmap, vmin=vmin, vmax=vmax)
    y_args = np.where(~np.isnan(y))
    P.scatter(
        y_args[0],
        y_args[1],
        s=0.5,
        c=y[~np.isnan(y)],
        cmap=cmap,
    )
    P.colorbar(orientation='horizontal')
    P.title(title)
