import matplotlib.pyplot as plt
import numpy as np

from src.utils import plot_hist_on_ax




def plot_uncertainty(ax, unc_name, uncs, unc_labels, **kwargs):

    plotters = {
        'sr': plot_sr,
        'vr': plot_vr,
        'pe': plot_pe,
        'mi': plot_mi,
    }
    assert unc_name in plotters.keys(), 'uncertainty not recognized'
    plotters[unc_name](ax, uncs, unc_labels, **kwargs)


def plot_general(ax, uncs, unc_labels, **kwargs):
    # plot_density_on_ax(ax, uncs_to_show, unc_labels, hist=True,)
    ax.set_xlabel('uncertainty')
    ax.set_ylabel('density')
    plot_hist_on_ax(
        ax,
        uncs,
        unc_labels,
        alpha=0.5,
        bins=30,
        stacked=True,
        density=True,
        **kwargs,
    )


def plot_sr(ax, uncs, unc_labels, **kwargs):
    plot_general(ax, uncs, unc_labels, **kwargs)


def plot_vr(ax, uncs, unc_labels, **kwargs):
    ax.set_ylabel('freq')
    ax.set_xlabel('uncertainty')
    for unc_label, unc in zip(unc_labels, uncs):
        unc, fq = np.unique(unc, return_counts=True)
        fq = fq / fq.sum()
        ax.plot(unc, fq, label=unc_label, **kwargs)


def plot_pe(ax, uncs, unc_labels, **kwargs):
    plot_general(ax, uncs, unc_labels, **kwargs)


def plot_mi(ax, uncs, unc_labels, **kwargs):
    plot_general(ax, uncs, unc_labels, **kwargs)
