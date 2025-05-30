from utils.my_random import MyPseudoRandom
from typing import List

class ConvLayer:
    def __init__(self, num_filters: int = 2, filter_size: int = 2, filter_depth: int = 3) -> None:
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filter_depth = filter_depth
        random = MyPseudoRandom()
        self.filters_matrix = random.generate_randn(filter_size, filter_size, filter_depth, num_filters)

        # Cache input for backward
        self.cache_last_input = None

    def iterate_possible_regions(self, image_matrix: List):
        h, w, c = len(image_matrix), len(image_matrix[0]), len(image_matrix[0][0])
        possible_regions = []
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                region = []
                for ki in range(self.filter_size):
                    row = []
                    for kj in range(self.filter_size):
                        pixel = image_matrix[i + ki][j + kj]
                        row.append(pixel)
                    region.append(row)
                possible_regions.append((region, i, j))
        return possible_regions
    
    def elementwise_multiply_and_sum(self, region_matrix, filter_matrix):
        total_sum = 0
        for ki in range(len(filter_matrix[0])):
            for kj in range(len(filter_matrix[1])):
                for c in range(len(region_matrix[ki][kj])):
                    total_sum += region_matrix[ki][kj][c] * filter_matrix[ki][kj][c]
        return total_sum

    def forward(self, img_matrix: List) -> List:
        self.cache_last_input = img_matrix

        h, w, c = len(img_matrix), len(img_matrix[0]), len(img_matrix[0][0])
        output = [[[0 for c in range(self.num_filters)] 
                   for j in range(w - self.filter_size + 1)] 
                for i in range(h - self.filter_size + 1)]
        
        # rebuild filter from init into the shape num_filters x filter_size x filter_size
        rebuild_filters_matrix = []
        for filter_idx in range(self.num_filters):
            current_filter = []
            for ki in range(self.filter_size):
                filter_row = []
                for kj in range(self.filter_size):
                    filter_pixel = []
                    for depth in range(self.filter_depth):
                        filter_pixel.append(self.filters_matrix[ki][kj][depth][filter_idx])
                    filter_row.append(filter_pixel)
                current_filter.append(filter_row)
            rebuild_filters_matrix.append(current_filter)

        # apply for each filter
        for filter_idx in range(len(rebuild_filters_matrix)):
            for img_region, i, j in self.iterate_possible_regions(img_matrix):
                conv = self.elementwise_multiply_and_sum(img_region, rebuild_filters_matrix[filter_idx])
                output[i][j][filter_idx] = conv

        return output
    
    def backward(self, d_out: List, learning_rate: float = 0.01) -> List:
        """
        Backward pass for convolution layer
        
        Args:
            d_out: Gradient of loss with respect to output of this layer
                   Shape: [output_h, output_w, num_filters]
        
        Returns:
            d_input: Gradient of loss with respect to output of previous layer
                    Shape: [input_h, input_w, input_c]
        """
        if self.cache_last_input is None:
            raise ValueError("Must call forward() before backward()")
        
        input_h, input_w, input_c = len(self.cache_last_input), len(self.cache_last_input[0]), len(self.cache_last_input[0][0])
        output_h, output_w = len(d_out), len(d_out[0])

        d_input = [[[0 for c in range(input_c)] 
                    for j in range(input_w)] 
                   for i in range(input_h)]
        
        d_filters = [[[[0 for f in range(self.num_filters)] 
                       for c in range(self.filter_depth)] 
                      for j in range(self.filter_size)] 
                     for i in range(self.filter_size)]
        
        # Compute gradients
        for filter_idx in range(self.num_filters):
            for out_i in range(output_h):
                for out_j in range(output_w):
                    # Get the gradient for this output position
                    grad_output = d_out[out_i][out_j][filter_idx]
                    
                    # Update filter gradients
                    for ki in range(self.filter_size):
                        for kj in range(self.filter_size):
                            for c in range(self.filter_depth):
                                input_i = out_i + ki
                                input_j = out_j + kj
                                d_filters[ki][kj][c][filter_idx] += grad_output * self.cache_last_input[input_i][input_j][c]

                    # Compute input gradients
                    for ki in range(self.filter_size):
                        for kj in range(self.filter_size):
                            for c in range(self.filter_depth):
                                input_i = out_i + ki
                                input_j = out_j + kj
                                d_input[input_i][input_j][c] += grad_output * self.filters_matrix[ki][kj][c][filter_idx]

        # update parameter
        for ki in range(self.filter_size):
            for kj in range(self.filter_size):
                for c in range(self.filter_depth):
                    for f in range(self.num_filters):
                        self.filters_matrix[ki][kj][c][f] -= learning_rate * d_filters[ki][kj][c][f]

        return d_input
