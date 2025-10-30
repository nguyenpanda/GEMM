from PyTest import CLITestCase, BufferReader

import numpy as np
import matplotlib.pyplot as plt
from nguyenpanda.swan import green, yellow

class SingleTest(CLITestCase):
    def _setup(self):
        super()._setup()
        cast = 'float64'
        self.objs['lhs'] = self.objs['lhs'].astype(cast)
        self.objs['rhs'] = self.objs['rhs'].astype(cast)
        self.objs['out'] = self.objs['out'].astype(cast)
        return self
    
    def logging_info(self):
        super().logging_info()
        print(green(f'{self.objs['lhs'].shape}'))
        print(green(f'{self.objs['rhs'].shape}'))
        print(green(f'{self.ufunc}'))
                
        print(yellow('Out'))
        print(self.objs['out'])
        print(yellow('Expected Output'))
        print(self.objs['expected_out'])
        
        print(f'         lhs[0, 0] = {self.objs['lhs'][0, 0]:.10f}')
        print(f'         rhs[0, 0] = {self.objs['rhs'][0, 0]:.10f}')
        print(f'         out[0, 0] = {self.objs['out'][0, 0]:.10f}')
        print(f'Expected out[0, 0] = {self.objs['expected_out'][0, 0]:.10f}')
        return self
    
    def plot_diff(self):
        diff = np.abs((self.objs['out'] - self.objs['expected_out']))
        num_error = np.sum(diff > self.cli_args.tolerance)
        
        if num_error == 0:
            return
        
        error = 100 * diff / np.abs(self.objs['expected_out'])
        error = error.flatten()
        N = error.size
        
        plt.figure(figsize=(8, 5))
        plt.hist(error, bins=100, color='skyblue', edgecolor='black')
        plt.title('Error Distribution Histogram')
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()

        return self

SingleTest() \
 	.run() \
	.logging_info()
    # .plot_diff()

