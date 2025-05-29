import math

def read_neural_network_file(file_path):
    x = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()                
            if line:
                x.append(int(line))
    return x[0], x[1:len(x)]

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_derivative(s):
    return s * (1 - s)

class MyPseudoRandom:
    def __init__(self, m = 7, a = 3, c = 3, seed = 5):
        """
        m: modulus
        a: multiplier
        c: increment
        X_init: seed, start value
        return pseudo random
        """
        self.m = m
        self.a = a
        self.c = c
        self.X = seed

    def next(self):
        self.X = (self.a * self.X + self.c) % self.m
        return self.X / self.m
    
class Neuron:
    def __init__(self, num_input, activation_function):
        self.weights = []
        self.bias = 1
        self.num_input = num_input
        self.activation_function = activation_function
        
    def init_random(self, random: MyPseudoRandom):
        for _ in range(self.num_input):
            self.weights.append(random.next())
        self.bias = random.next()

    def compute_z(self, inputs):
        return sum(w * i for w, i in zip(self.weights, inputs)) + self.bias

    def activation(self, inputs):
        z = self.compute_z(inputs)
        return self.activation_function(z)


class Layer:
    def __init__(self, num_input, num_neuron, activation_function):
        self.neurons = []
        for _ in range(num_neuron):
            self.neurons.append(Neuron(num_input, activation_function))

    def initilize_ramdon_weights(self, random: MyPseudoRandom):
        for neuron in self.neurons:
            neuron.init_random(random)

    def forward(self, inputs):
        return [neuron.activation(inputs) for neuron in self.neurons]


class MyNN:
    def __init__(self, number_of_layers: int, number_of_nerons: list, activation_function):
        self.__layers = []
        for i in range(number_of_layers - 1):
            print("Layer", i, "with", number_of_nerons[i], "input and", number_of_nerons[i + 1], "neurons")
            layer = Layer(number_of_nerons[i], number_of_nerons[i + 1], activation_function)
            self.__layers.append(layer)
    
    def initilize_weights_random(self, random: MyPseudoRandom):
        for layer in self.__layers:
            layer.initilize_ramdon_weights(random)

    # def feed_forward(self, input_data):
    #     output_data = input_data
    #     for layer in self.__layers:
    #         output_data = layer.forward(output_data)
    #     return output_data

    def feed_forward(self, input_data):
        activations = [input_data]
        zs = []
        current_input = input_data

        for layer in self.__layers:
            z_layer = []
            a_layer = []
            for neuron in layer.neurons:
                z = neuron.compute_z(current_input)
                a = neuron.activation_function(z)
                z_layer.append(z)
                a_layer.append(a)
            zs.append(z_layer)
            activations.append(a_layer)
            current_input = a_layer
        
        print("zs", zs)
        print("activations", activations)
        print("length zs", len(zs))
        print("legth activations", len(activations))

        return zs, activations
    
    def backpropagation(self, input_data, target_output, learning_rate):
        zs, activations = self.feed_forward(input_data)
        
        delta = [0] * len(self.__layers)

        L = len(self.__layers) - 1
        delta[L] = []

        for i, neuron in enumerate(self.__layers[L].neurons):
            a = activations[L + 1][i]
            z = zs[L][i]
            error = a - target_output[i]
            delta



file_path = "./neural_network1.txt"
number_layers, number_of_neurons = read_neural_network_file(file_path)
input_vector = [0.5 for _ in range(number_of_neurons[0])]

nn = MyNN(number_layers, number_of_neurons, sigmoid)
random = MyPseudoRandom(seed=12)
nn.initilize_weights_random(random)
zs, activations = nn.feed_forward(input_vector)
