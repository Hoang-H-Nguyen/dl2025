from typing import List, Tuple, Union
import math

def shape(matrix: Union[List, float]) -> Tuple[int, ...]:
    if isinstance(matrix, list):
        outermost_size = len(matrix)
        row_shape = shape(matrix[0])
        return (outermost_size, *row_shape)
    else:
        # No more dimensions, so we're done
        return ()

def matrix_len(*shape: tuple) -> int:
    return math.prod(shape)
