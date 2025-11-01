from PyPerf import BM_JsonParser

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider
from matplotlib.colors import PowerNorm

import re
import argparse

parser = argparse.ArgumentParser(description="A sample script demonstrating argparse.")
parser.add_argument("file", type=str, help="JSON file")
args = parser.parse_args()

def plot_heat_map():
    col_max = np.max(real_time, axis=0)
    ratio = col_max / real_time

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        pd.DataFrame(ratio, columns=N, index=T),
        annot=True, fmt='.2f', cmap='coolwarm',
        ax=ax,
        norm=PowerNorm(gamma=0.4),
    )
    ax.set_title('Relative Speedup per Matrix Size for Fork-Join Element-wise Addition')
    ax.set_xlabel('N')
    ax.set_ylabel('Threshold')
    fig.savefig(BM_Parser.img_dir / f'{BM_Parser.json_file.stem}-speedup_ratio.png')
    
    plt.show()
    
def plot_3d():
    col_max = np.max(real_time, axis=0)
    ratio = col_max / real_time
    X, Y = np.meshgrid(N, T) # type: ignore
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X, Y, ratio,
        cmap='coolwarm',
        linewidth=0,
        antialiased=True,
        norm=PowerNorm(gamma=0.5),
    )

    ax.set_xlabel('Matrix Size (N)')
    ax.set_ylabel('Threshold')
    ax.set_zlabel('Relative Speedup')
    ax.set_title('Relative Speedup Surface — Fork-Join Element-wise Addition')

    fig.colorbar(surf, shrink=0.5, aspect=10)

    fig.savefig(BM_Parser.img_dir / f'{BM_Parser.json_file.stem}-speedup_ratio_3D.png')

    ax_elev = plt.axes((0.25, 0.02, 0.50, 0.02))
    ax_azim = plt.axes((0.25, 0.05, 0.50, 0.02))
    
    slider_elev = Slider(ax_elev, 'Elev', 0, 90, valinit=90)
    slider_azim = Slider(ax_azim, 'Azim', 0, 360, valinit=90)

    def update(val):
        ax.view_init(elev=slider_elev.val, azim=slider_azim.val)
        fig.canvas.draw_idle()

    slider_elev.on_changed(update)
    slider_azim.on_changed(update)
    plt.show()

if __name__ == '__main__':
    BM_Parser = BM_JsonParser(args.file)
    
    BM_Parser.print_context([
        'date',
        'host_name',
        'load_avg',
		'OMP_ENABLE',
    	'OMP_NUM_THREADS',
    	'PRECISION_STATUS',
    	'PRECISION_MODE',
     	'MATMUL_ORDER',
	])
    
    BM_Parser.parse([
        'real_time',
        'cpu_time',
        'iterations',
        'time_unit',
        'N'
	])
    
    d = BM_Parser.benchmarks
    
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
    N = np.array(N)
    real_time = np.array(real_time)
    
    print(real_time.transpose())
    print('real_time.shape:', real_time.shape)
    print('              N:', N)
    print('      threshold:', T)
    
    plot_heat_map()
    plot_3d()
    