from PyPerf import BM_JsonParser

import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="A sample script demonstrating argparse.")
parser.add_argument("file", type=str, help="JSON file")
args = parser.parse_args()

def plot_execute_time():
	fig, axes = plt.subplots(len(Y_FEATURES), 1, figsize=(8, 8))
	fig.suptitle('Average execution time', fontsize=16)
	fig.supxlabel('N')
 
	for ax, feature in zip(axes, Y_FEATURES):
		for i, (BM_name, d) in enumerate(BM_Parser.benchmarks.items()):
			ax.plot(d['N'], d[feature], label=BM_name)
			ax.set_title(feature)
			ax.set_ylabel('time')
			ax.grid(True, which="both", ls="--")
			ax.legend(fontsize=8)
	plt.savefig(BM_Parser.img_dir / f'{BM_Parser.json_file.stem}-execution_time.png')
 
def plot_ratio():
	fig, axes = plt.subplots(len(Y_FEATURES), 1, figsize=(8, 8))
	fig.suptitle('Speedup ratio', fontsize=16)
	fig.supxlabel('N')
	BM_Baseline = list(BM_Parser.benchmarks.keys())[0]

	for ax, feature in zip(axes, Y_FEATURES):
		for i, (BM_name, d) in enumerate(BM_Parser.benchmarks.items()):
			if i == BM_Baseline: continue
			ax.plot(d['N'], np.array(BM_Parser.benchmarks[BM_Baseline][feature]) / np.array(d[feature]), label=BM_name)
			ax.set_xscale('log', base=2)
			ax.set_title(feature)
			ax.set_ylabel('ratio')
			ax.grid(True, which="both", ls="--")
			ax.legend(fontsize=8)
	plt.savefig(BM_Parser.img_dir / f'{BM_Parser.json_file.stem}-speedup_ratio.png')

if __name__ == '__main__':
    BM_Parser = BM_JsonParser(args.file)
    BM_Parser.print_context([
        'date',
        'host_name',
        'load_avg',
		'ADD_RECURSIVE_THRESHOLD',
    	'MATMUL_ORDER',
    	'MUL_RECURSIVE_THRESHOLD',
    	'OMP_ENABLE',
     	'OMP_NUM_THREADS',
    	'PRECISION_MODE',
    	'PRECISION_STATUS',
	])
    
    BM_Parser.parse([
        'real_time',
        'cpu_time',
        'iterations',
        'time_unit',
        'N'
	])
    
    Y_FEATURES = ['real_time', 'cpu_time']
    
    plot_execute_time()
    
    plot_ratio()
    
    plt.show()