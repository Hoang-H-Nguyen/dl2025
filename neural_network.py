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

    def activation(self, inputs):
        z = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
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
            layer = Layer(number_of_nerons[i], number_of_nerons[i + 1], activation_function)
            self.__layers.append(layer)
    
    def initilize_weights_random(self, random: MyPseudoRandom):
        for layer in self.__layers:
            layer.initilize_ramdon_weights(random)

    def feed_forward(self, input_data):
        output_data = input_data
        for layer in self.__layers:
            output_data = layer.forward(output_data)
        return output_data


file_path = "./neural_network1.txt"
number_layers, number_of_neurons = read_neural_network_file(file_path)
input_vector = [0.5 for _ in range(number_of_neurons[0])]

nn = MyNN(number_layers, number_of_neurons, sigmoid)
random = MyPseudoRandom(seed=12)
nn.initilize_weights_random(random)
output = nn.feed_forward(input_vector)
print("Output of the neural network:", output)