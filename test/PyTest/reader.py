import numpy as np
import struct

class BufferReader:
    str_to_np_dtype = {
		"float32": np.float32,
		"float64": np.float64,
		"int32": np.int32,
		"int64": np.int64,
		"uint8": np.uint8,
	}
    
    @classmethod
    def read_buffer(cls, filename):
        with open(filename, 'rb') as f:
            magic = f.read(8)
            if magic != b'BUFFER1\0':
                raise ValueError('Invalid file format or magic header mismatch')
            
            dtype_len = struct.unpack("B", f.read(1))[0]
            dtype = f.read(dtype_len).decode()
            rdim = struct.unpack("Q", f.read(8))[0]
            cdim = struct.unpack("Q", f.read(8))[0]
            
            np_dtype = cls.str_to_np_dtype.get(dtype)
            if np_dtype is None:
                raise ValueError(f'Unsupported dtype: {dtype}')
            
            data = np.fromfile(f, dtype=np_dtype, count=rdim * cdim)
        return data.reshape((rdim, cdim))
        
    @classmethod
    def get_buffer_from_csv_string(cls, string_buffer: str) -> np.ndarray:
        return np.array([
			[float(x) for x in n.split() if x not in ('', '\n')]
			for n in string_buffer.strip().split('\n')
			if n.strip()
		])
