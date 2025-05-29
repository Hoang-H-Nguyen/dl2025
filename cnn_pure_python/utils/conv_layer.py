from utils.my_random import MyPseudoRandom
from typing import List

class ConvLayer:
    def __init__(self, num_filters: int = 2, filter_size: int = 2, filter_depth: int = 3) -> None:
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filter_depth = filter_depth
        random = MyPseudoRandom()
        self.filters_matrix = random.generate_randn(filter_size, filter_size, filter_depth, num_filters)

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
    
    def backward(self):
        pass

