from utils.my_random import MyPseudoRandom
import math

class DenseLayer:
    
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu') -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        random = MyPseudoRandom()
        self.weights = [[random.next() for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [random.next() for _ in range(output_size)]

    def flatten(self, input_matrix):
        if isinstance(input_matrix[0], list):
            # check h, w, c image
            if isinstance(input_matrix[0][0], list):
                flattened = []
                for i in range(len(input_matrix)):
                    for j in range(len(input_matrix[0])):
                        for k in range(len(input_matrix[0][0])):
                            flattened.append(input_matrix[i][j][k])
                return flattened
            # check h, w image
            else:
                flattened = []
                for row in input_matrix:
                    flattened.extend(row)
                return flattened
        else:
            # already 1d matrix
            return input_matrix
        
    def apply_activation(self, x):
        if self.activation == 'relu':
            return max(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + math.exp(-max(-500, min(500, x))))
        elif self.activation == 'tanh':
            return math.tanh(x)
        elif self.activation == 'softmax':
            # For softmax, we need all values, so this will be handled separately
            return x
        else:  # linear
            return x

    def apply_softmax(self, outputs):
        max_val = max(outputs)
        exp_outputs = [math.exp(x - max_val) for x in outputs]
        sum_exp = sum(exp_outputs)
        return [x / sum_exp for x in exp_outputs]

    def forward(self, input_data):
        flatten_input = self.flatten(input_data)

        z = []
        for j in range(self.output_size):
            weighted_sum = 0
            for i in range(self.input_size):
                weighted_sum = flatten_input[i] * self.weights[j][i]
            output = weighted_sum + self.biases[j]
            z.append(output)

        if self.activation == 'softmax':
            return self.apply_softmax(z)
        else:
            return [self.apply_activation(z_i) for z_i in z]
