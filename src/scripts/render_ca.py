from argparse import ArgumentParser

from ca import CA, render

def main():
    args = parser.parse_args()
    args = dict(vars(args).items())
    
    ca_num = args['ca_num'][0]
    
    ca_nhd = args['neighborhood']
    ca_nhd = tuple(ca_nhd)
    
    ca_states = args['states'][0]
    input_seq = args['in_seq']
    
    proc_steps = args['steps'][0]
    
    pre_pad_seq = args.get('pre_pad')
    if pre_pad_seq is None:
        pre_pad_seq = False
    else:
        pre_pad_seq = pre_pad_seq > 0
        
    post_pad_seq = args.get('post_pad')
    if post_pad_seq is None:
        post_pad_seq = False
    else:
        post_pad_seq = post_pad_seq > 0
        
    plot_cmap = args['cmap']
    if isinstance(plot_cmap, list):
        plot_cmap = plot_cmap[0]
    
    out_fpath = args['output'][0]
    
    ca = CA(rule=ca_num, nhd=ca_nhd, n_states=ca_states)
    
    render(
        out_fpath=out_fpath,
        automaton=ca,
        in_seq=input_seq,
        steps=proc_steps,
        pre_pad=pre_pad_seq,
        post_pad=post_pad_seq,
        cmap_name=plot_cmap,
    )

parser = ArgumentParser(
    "render_ca",
    description="compute and render the evolution of a 1D cellular automaton"
)

parser.add_argument(
    "-C",
    "--ca-num",
    dest="ca_num",
    action="store",
    nargs=1,
    type=int,
    required=True,
    help="output filepath"
)
parser.add_argument(
    "-o",
    "--output",
    action="store",
    nargs=1,
    type=str,
    required=True,
    help="output filepath"
)
parser.add_argument(
    "-n",
    "--neighborhood",
    action="store",
    nargs=2,
    type=int,
    required=True,
    help="left and right neighborhood sizes"
)
parser.add_argument(
    "-S",
    "--states",
    action="store",
    nargs=1,
    type=int,
    required=True,
    help="Wolfram number of the cellular automaton rule"
)
parser.add_argument(
    "-s",
    "--steps",
    action="store",
    nargs=1,
    type=int,
    required=True,
    help="number of steps to compute"
)
parser.add_argument(
    "-p",
    dest="pre_pad",
    action="count",
    help="pad inputs to produce fixed length results"
)
parser.add_argument(
    "-P",
    dest="post_pad",
    action="count",
    help="pad outputs to produce fixed length results"
)
parser.add_argument(
    "-c",
    "--cmap",
    action="store",
    nargs=1,
    type=str,
    default="hot",
    required=False,
    help="name of matplotlib colormap to use in plot"
)
parser.add_argument(
    "in_seq",
    action="store",
    nargs="+",
    type=int,
    help="initial state as integers, whitespace separated"
)
