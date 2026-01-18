
import json
import re
from pathlib import Path

from nguyenpanda.swan import green, yellow

class BM_JsonParser:

    def __init__(self, file: str | Path, img_dir: str = 'img'):
        self.json_file: Path = Path(file)
        self.benchmarks: dict[str, dict] = dict()

        try:
            with open(self.json_file) as f:
                self.raw_data: dict = json.load(f)
            self.data_context = self.raw_data['context']
            self.data_benchmarks = self.raw_data['benchmarks']
            self.img_dir = self.json_file.parents[1] / img_dir
            self.img_dir.mkdir(exist_ok=True)
        except FileExistsError as e:
            raise e
    
    def print_context(self, attributes: list | tuple | set):
        print(green('=' * 20))
        for attr in attributes:
            print(f'{attr}: {yellow(self.data_context.get(attr))}')
        print(green('=' * 20))
        
    def parse(self, keys: list | tuple | set):
        for b in self.data_benchmarks:
            BM_name = re.search(r'(BM_[^/]+)', b['name']).group(1) # type: ignore
            BM_size = int(re.search(r"/(\d+)/", b['name']).group(1)) # type: ignore
            
            if BM_name not in self.benchmarks.keys():
                self.benchmarks[BM_name] = {k: [] for k in keys}
            
            for k in keys:
                if k != 'N':
                    self.benchmarks[BM_name][k].append(b[k])
                else:
                    self.benchmarks[BM_name]['N'].append(BM_size)
    