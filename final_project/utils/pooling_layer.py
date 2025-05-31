from typing import List

class MaxPoolingLayer:
    def __init__(self, pooling_size: int = 2):
        self.pooling_size = pooling_size

        # Cache for backpropagation - stores which positions had max values
        self.cached_max_positions = None  # Will store (input_i, input_j) for each output position
        self.cached_input_shape = None
        self.cached_output_shape = None

    def iterate_possible_regions(self, image_matrix: List) -> List:
        h, w, c = len(image_matrix), len(image_matrix[0]), len(image_matrix[0][0])
        pooling_height = h // self.pooling_size
        pooling_width = w // self.pooling_size

        possible_regions = []
        for i in range(0, pooling_height * self.pooling_size, self.pooling_size):
            for j in range(0, pooling_width * self.pooling_size, self.pooling_size):
                region = []
                for pi in range(self.pooling_size):
                    row = []
                    for pj in range(self.pooling_size):
                        row.append(image_matrix[i + pi][j + pj])
                    region.append(row)
                possible_regions.append((region, i // self.pooling_size, j // self.pooling_size))
        return possible_regions
    
    def forward(self, image_matrix: List) -> List:
        h, w, c = len(image_matrix), len(image_matrix[0]), len(image_matrix[0][0])
        self.cached_input_shape = (h, w, c)
        
        output_h = h // self.pooling_size
        output_w = w // self.pooling_size
        self.cached_output_shape = (output_h, output_w, c)

        output = [[[0 for cc in range(c)] 
                   for j in range(output_w)] 
                for i in range(output_h)]
        self.cached_max_positions = [[[None for _ in range(c)] 
                              for j in range(output_w)] 
                             for i in range(output_h)]
        
        for region, out_i, out_j in self.iterate_possible_regions(image_matrix):
            for dc in range(c):
                max_val = float('-inf')
                max_pos = (0,0)

                for di in range(self.pooling_size):
                    for dj in range(self.pooling_size):
                        pixel_val = region[di][dj][dc]
                        if pixel_val > max_val:
                            max_val = pixel_val
                            max_pos = (di, dj)
                
                # Store output value     
                output[out_i][out_j][dc] = max_val

                # Store the input coordinates of the maximum value
                input_i = out_i * self.pooling_size + max_pos[0]
                input_j = out_j * self.pooling_size + max_pos[1]
                self.cached_max_positions[out_i][out_j][dc] = (input_i, input_j)

        return output
    
    def backward(self, d_output: List) -> List:
        h, w, c = self.cached_input_shape
        output_h, output_w, _ = self.cached_output_shape

        if (len(d_output) != output_h or len(d_output[0]) != output_w or len(d_output[0][0]) != c):
            raise ValueError(f"d_output shape {len(d_output)}x{len(d_output[0])}x{len(d_output[0][0])} "
                           f"doesn't match expected output shape {output_h}x{output_w}x{c}")
        
        d_input = [[[0.0 for cc in range(c)] 
                    for j in range(w)] 
                   for i in range(h)]
        
        for out_i in range(output_h):
            for out_j in range(output_w):
                for ch in range(c):
                    max_input_i, max_input_j = self.cached_max_positions[out_i][out_j][ch]
                    d_input[max_input_i][max_input_j][ch] += d_output[out_i][out_j][ch]
        return d_input
    
    def get_max_positions_info(self):
        """for trace log"""
        if self.cached_max_positions is None:
            return None
        
        info = {
            'input_shape': self.cached_input_shape,
            'output_shape': self.cached_output_shape,
            'max_positions': self.cached_max_positions
        }
        return info
