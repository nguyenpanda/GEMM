from PyPerf import BM_JsonParser
from PyPerf import BM_CliParser as BM_CliBase

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Rectangle, Patch
from matplotlib.widgets import Slider
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D

class BM_CliParser(BM_CliBase):
    
    @classmethod
    def setup(cls, parser):
        parser = super().setup(parser)
        parser.add_argument("--no_patch", action='store_true', help="Display maximum value per row")
        parser.add_argument("--no_seq", action='store_true', help="Remove sequence value (Threshold > N)")
        parser.add_argument("--unique_seq", action='store_true', help="Only 1 sequence value on heatmap")
        return parser


args = BM_CliParser().args

def plot_heat_map():
    col_max = real_time_df.max(axis=0)
    ratio = col_max / real_time_df
    
    mask = ratio.index.to_numpy()[:, None] >= ratio.columns.to_numpy()[None, :]
    if args.no_seq:
        ratio[mask] = 0.0
        
    if args.unique_seq:
        unique_seq_mask = ratio.index.to_numpy()[:, None] >= (ratio.columns.to_numpy()[None, :] * 2)
        ratio[unique_seq_mask] = 0.0
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        ratio, ax=ax,
        annot=True, fmt='.2f', cmap='coolwarm',
        norm=PowerNorm(gamma=0.4),
    )
    
    # Sequence
    row_idx, col_idx = np.where(mask)
    row_val = np.unique(row_idx)
    col_val = np.unique(col_idx)
    for i, j in zip(row_val, col_val):
        ax.plot((j, i), (j+1, i), 'black', linewidth=1)
        ax.plot((j+1, i), (j+1, i+1), 'black', linewidth=1)
    
    if not args.no_patch:
        def _max_value(df, color, lw):
            for col_idx, row_idx in enumerate(
                df.index.get_loc(r) 
                for r in df.idxmax(axis=0)
            ):
                rect = Rectangle(
                    (col_idx, row_idx), 1, 1, # type: ignore
                    fill=False,
                    edgecolor=color,
                    lw=lw,
                )
                ax.add_patch(rect)
        # Maximum Value Global
        temp = ratio.copy()
        _max_value(ratio, 'yellow', 4)
        
        # Maximun Value Sequence
        temp[mask] = 0
        _max_value(temp, 'green', 2)
        
    ax.set_title(f'Relative Speedup per Matrix Size (N) for Fork-Join {BM_Parser.json_file.stem.split('-')[0]}')
    if args.legend:
        box = ax.get_position()
        ax.set_position((box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9))
        ax.legend(handles=[
            Line2D([0], [0], color='black', lw=4, label='N <= Threshold'),
            Patch(linewidth=2, edgecolor='yellow', facecolor='white', label='Relative maximum per N'),
            Patch(linewidth=2, edgecolor='green', facecolor='white', label='Relative maximun per N only for N >= Threshold'),
            ], 
            loc='upper center', bbox_to_anchor=(0.5, -0.15),
            ncol=3, fancybox=True, shadow=True
        )
    fig.savefig(BM_Parser.img_dir / f'{BM_Parser.json_file.stem}-heatmap.png')


if __name__ == '__main__':
    BM_Parser = BM_JsonParser(args.file)
    
    BM_Parser.print_context([
        'date',
        'host_name',
        'load_avg',
        'MPI_COMM_SIZE',
        'node_names',
        'processes',
        'strategy',
	])
    
    BM_Parser.parse([
        'real_time',
        'cpu_time',
        'iterations',
        'time_unit',
        'N'
	])
    
    d = BM_Parser.benchmarks
    
    import re
    T, N, real_time = list(), None, list()
    for plot, val in d.items():
        if N is None:
            N = d[plot]['N']
        else:
            if N != val['N']:
                raise ValueError
        
        T.append(int(re.search(r"_T(\d+)$", plot).group(1))) # type: ignore
        real_time.append(d[plot]['real_time'])
    
    T = np.array(T)
    idxT = T.argsort()
    T = T[idxT]
    
    N = np.array(N)
    idxN = N.argsort()
    N = N[idxN]
    
    # real_time_df = pd.DataFrame(
    #     np.array(real_time)[idxT][:, idxN], 
    #     columns=N, index=T,
    # ) \
    #     .rename_axis('Threshold', axis='index') \
    #     .rename_axis('N', axis='columns')
        
    # pd.set_option('display.precision', 2)
    # if args.display_console:
    #     print(real_time_df)
    
    # plot_heat_map()
    
    # if args.show_img:
    #     plt.show()
    # plt.close()
        