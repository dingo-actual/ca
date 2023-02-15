from os import PathLike
from typing import List

from numpy import array
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from .ca import CA


def render(
    out_fpath: PathLike,
    automaton: CA, 
    in_seq: List[int], 
    steps: int, 
    pre_pad: bool, 
    post_pad: bool,
    cmap_name: str = "hot",
    **kwargs
) -> None:
    """Render the state evolution history of a cellular automaton object and save to file.

    Args:
        out_fpath (PathLike): Path to save the rendered output.
        automaton (CA): Cellular automaton object.
        in_seq (List[int]): Initial state sequence to render.
        steps (int): Number of steps to compute.
        pre_pad (bool): Pad zeros to inputs to produce results with fixed length.
        post_pad (bool): Pad zeros to outputs to produce results with fixed length.
        cmap_name (str, optional): Name of the matplotlib colormap used to color outputs. Defaults to "hot".
    """
    cmap = get_cmap(cmap_name)
    
    # compute the evolution of the automaton, then convert states into floats between 0. and 1.
    # with 1. being state 0 and 0. being state <self.n_states> - 1
    steps_arr = (
        1.0 - 
        array(automaton(in_seq, steps, pre_pad, post_pad), order="C", ndmin=2) / 
        (automaton.n_states - 1)
    )
    
    # each value in <steps_arr> is a single pixel value, so any array smaller than
    # 256 by 256 will be difficult to see
    # if either dimension of the history is less than 256, repeat the values until
    # the dimensions are at least 256 by 256
    for dim in [0, 1]:
        if steps_arr.shape[dim] < 256:
            reps = 256 // steps_arr.shape[dim] + 1
            steps_arr = steps_arr.repeat(reps, axis=dim)
    
    # plot the results above
    fig, ax = plt.subplots()
    ax.imshow(steps_arr, cmap=cmap, interpolation="none")
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.spines[["right", "left", "top", "bottom"]].set_visible(False)
    
    # save plot to disk and cleanup
    plt.savefig(out_fpath, **kwargs)
    plt.close(fig)
