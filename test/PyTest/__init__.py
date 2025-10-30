from .reader import BufferReader

import argparse
import numpy as np
from pathlib import Path
from typing import Callable, Optional
from nguyenpanda.swan import green, red

def get_latest_buffer_dir(base_dir='buffer'):
    base_path = Path(base_dir)
    if not base_path.exists() or not base_path.is_dir():
        return None

    dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not dirs:
        return None

    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return str(dirs[0])

class Mapper:

    np_operator = {
		'+': np.add,
		'-': np.subtract,
  		'*': np.multiply,
  		'/': np.divide,
		'@': np.matmul,
	}

class CLITestCase:
    
    def __init__(self):
        self.argument_parser = self._init_argparse()
        self.cli_args = self.argument_parser.parse_args()
        
        self.dir_path = Path(self.cli_args.dir)
        operator_file = self.dir_path / 'operator.txt'
        if not operator_file.exists():
            raise FileNotFoundError(f'Missing operator.txt in {self.dir_path}')

        with open(operator_file, 'r') as f:
            op_symbol = f.read().strip()

        try:
            self.ufunc = Mapper.np_operator[op_symbol]
        except ValueError as e:
            raise ValueError(f'Unknown operator \'{op_symbol}\' in operator.txt')
        
        self.objs = dict()
        
    def _setup(self):
        self.objs['lhs'] = BufferReader.read_buffer(self.dir_path / 'lhs.nguyenpanda')
        self.objs['rhs'] = BufferReader.read_buffer(self.dir_path / 'rhs.nguyenpanda')
        self.objs['out'] = BufferReader.read_buffer(self.dir_path / 'out.nguyenpanda')
        return self
        
    def run(self):
        self._setup()
        lhs = self.objs['lhs']
        rhs = self.objs['rhs']
        out = self.objs['out']
        
        try:
            expected_out = self.ufunc(lhs, rhs)
            diff = np.abs(out - expected_out)
        except Exception as e:
            raise e
        
        num_error = np.sum(diff > self.cli_args.tolerance)
        if num_error == 0:
            print(green(f'Correct answer (num_error = {num_error})'))
        else:
            print(red(f'Wrong answer (num_error = {num_error})'))
            
        self.objs['expected_out'] = expected_out
        return self
    
    def logging_info(self):
        print(self.cli_args)
        return self

    def _init_argparse(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dir', default=get_latest_buffer_dir(), help='directory contains `lhs.bin`, `rhs.bin`, `out.bin`')
        parser.add_argument('--tolerance', default=1e-5, type=float, help=f'test operator {Mapper.np_operator}')
        return parser