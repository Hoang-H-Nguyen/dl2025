from typing import List
import math

class MaxPooling:
    def __init__(self, pooling_size: int = 2):
        self.pooling_size = pooling_size

    def iterate_possible_regions(self, image_matrix: List) -> List:
        h, w, c = len(image_matrix), len(image_matrix[0]), len(image_matrix[0][0])
        pooling_height = math.floor(h - self.pooling_size + 1)
        pooling_width = math.floor(w - self.pooling_size + 1)

        possible_regions = []
        for i in range(pooling_height):
            for j in range(pooling_width):
                region = []
                for pi in range(self.pooling_size):
                    row = []
                    for pj in range(self.pooling_size):
                        row.append(image_matrix[i + pi][j + pj])
                    region.append(row)
                possible_regions.append((region, i, j))
        return possible_regions
    
    def forward(self, image_matrix: List) -> List:
        h, w, c = len(image_matrix), len(image_matrix[0]), len(image_matrix[0][0])
        output_h = math.floor(h - self.pooling_size + 1)
        output_w = math.floor(w - self.pooling_size + 1)

        output = [[[0 for c in range(c)] 
                   for j in range(output_w)] 
                for i in range(output_h)]
        
        for region, i, j in self.iterate_possible_regions(image_matrix):
            for dc in range(c):
                max_val = float('-inf')
                for di in range(self.pooling_size):
                    for dj in range(self.pooling_size):
                        pixel_val = region[di][dj][dc]
                        if pixel_val > max_val:
                            max_val = pixel_val
                output[i][j][dc] = max_val
        return output

# Test the implementation
image = [
    [[1,2,3],    [4,5,6],    [7,8,9]],
    [[16,17,18], [19,20,21], [22,23,24]],
    [[31,32,33], [34,35,36], [37,38,39]],
]

maxpooling = MaxPooling()
result = maxpooling.forward(image)
print(result)