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

        # Cache variables for backpropagation
        self.cached_input_data = None
        self.cached_flatten_input = None
        self.cached_z = None
        self.cached_activations = None
        self.cached_input_shape = None

    @property
    def value(self):
        return self.input_size

    @value.setter
    def value(self, input_size):
        self.input_size = input_size

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
        
    def reshape_to_original(self, flattened_data, original_input):
        """Reshape flattened data back to the original input shape"""
        if isinstance(original_input[0], list):
            # check h, w, c image
            if isinstance(original_input[0][0], list):
                h, w, c = len(original_input), len(original_input[0]), len(original_input[0][0])
                reshaped = [[[0 for _ in range(c)] for _ in range(w)] for _ in range(h)]
                idx = 0
                for i in range(h):
                    for j in range(w):
                        for k in range(c):
                            reshaped[i][j][k] = flattened_data[idx]
                            idx += 1
                return reshaped
            # check h, w image
            else:
                h, w = len(original_input), len(original_input[0])
                reshaped = [[0 for _ in range(w)] for _ in range(h)]
                idx = 0
                for i in range(h):
                    for j in range(w):
                        reshaped[i][j] = flattened_data[idx]
                        idx += 1
                return reshaped
        else:
            # already 1d matrix
            return flattened_data
        
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
    
    def activation_derivative(self, x, activation_output=None):
        if self.activation == 'relu':
            return 1.0 if x > 0 else 0.0
        elif self.activation == 'sigmoid':
            if activation_output is not None:
                return activation_output * (1 - activation_output)
            else:
                sig = 1 / (1 + math.exp(-max(-500, min(500, x))))
                return sig * (1 - sig)
        elif self.activation == 'tanh':
            if activation_output is not None:
                return 1 - activation_output ** 2
            else:
                tanh_val = math.tanh(x)
                return 1 - tanh_val ** 2
        elif self.activation == 'softmax':
            # Softmax derivative is handled separately in backprop
            return 1.0
        else:  # linear
            return 1.0

    def forward(self, input_data):
        self.cached_input_data = input_data
        self.cached_input_shape = input_data

        flatten_input = self.flatten(input_data)
        self.cached_flatten_input = flatten_input

        z = []
        for j in range(self.output_size):
            weighted_sum = 0
            for i in range(self.input_size):
                weighted_sum += flatten_input[i] * self.weights[j][i]
            output = weighted_sum + self.biases[j]
            z.append(output)

        self.cached_z = z

        if self.activation == 'softmax':
            activations =  self.apply_softmax(z)
        else:
            activations = [self.apply_activation(z_i) for z_i in z]

        self.cached_activations = activations
        return activations
        
    def backward(self, d_output, learning_rate=0.01):
        if self.cached_flatten_input is None or self.cached_z is None:
            raise ValueError("Forward pass must be called before backward pass")
        # Step 1: Compute gradient with respect to z
        if self.activation == 'softmax':
            d_z = d_output[:]
        else:
            # For other activations: d_z = d_output * activation'(z)
            d_z = []
            for i in range(self.output_size):
                activation_grad = self.activation_derivative(
                    self.cached_z[i], 
                    self.cached_activations[i]
                )
                d_z.append(d_output[i] * activation_grad)

        d_weights = []
        d_biases = []

        # Step 2: Compute gradients with respect to weights and biases
        d_weights = []
        d_biases = []
        
        for j in range(self.output_size):
            # Gradient w.r.t. weights and biases: d_weight = d_z * input and d_bias = d_z
            d_biases.append(d_z[j])
            d_weight_row = []
            for i in range(self.input_size):
                d_weight_row.append(d_z[j] * self.cached_flatten_input[i])
            d_weights.append(d_weight_row)

        # Step 3: Compute gradient with respect to input
        d_input_flat = [0.0] * self.input_size
        for i in range(self.input_size):
            for j in range(self.output_size):
                d_input_flat[i] += d_z[j] * self.weights[j][i]

        # Step 4: Update weights and biases using gradients
        for j in range(self.output_size):
            self.biases[j] -= learning_rate * d_biases[j]
            for i in range(self.input_size):
                self.weights[j][i] -= learning_rate * d_weights[j][i]

        # Step 5: Reshape d_input back to original input shape
        d_input = self.reshape_to_original(d_input_flat, self.cached_input_shape)
        
        return d_input
    
    def get_gradients(self, d_output):
        """
        The same as backward without really update w, b
        """
        if self.cached_flatten_input is None or self.cached_z is None:
            raise ValueError("Forward pass must be called before computing gradients")
        
        if self.activation == 'softmax':
            d_z = d_output[:]
        else:
            d_z = []
            for i in range(self.output_size):
                activation_grad = self.activation_derivative(
                    self.cached_z[i], 
                    self.cached_activations[i]
                )
                d_z.append(d_output[i] * activation_grad)
        
        d_weights = []
        d_biases = []
        
        for j in range(self.output_size):
            d_biases.append(d_z[j])
            d_weight_row = []
            for i in range(self.input_size):
                d_weight_row.append(d_z[j] * self.cached_flatten_input[i])
            d_weights.append(d_weight_row)
        
        d_input_flat = [0.0] * self.input_size
        for i in range(self.input_size):
            for j in range(self.output_size):
                d_input_flat[i] += d_z[j] * self.weights[j][i]
        
        # Reshape d_input back to original input shape
        d_input = self.reshape_to_original(d_input_flat, self.cached_input_shape)
        
        return {
            'weights': d_weights,
            'biases': d_biases,
            'input': d_input
        }
