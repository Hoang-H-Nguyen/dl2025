import math
import matplotlib.pyplot as plt
from utils.conv_layer import ConvLayer
from utils.pooling_layer import MaxPoolingLayer
from utils.dense_layer import DenseLayer
import utils.shape_util
import utils.image_util
import numpy as np

def cross_entropy(predicted_probs, true_label):
    return -math.log(predicted_probs[true_label] + 1e-15)


def forward(image_matrix: list, label):
    conv = ConvLayer(filter_size= 32, num_filters=4)
    max_pool = MaxPoolingLayer(pooling_size=2)
    out = conv.forward(image_matrix)
    out = max_pool.forward(out)

    input_shape = utils.shape_util.shape(out)
    dense = DenseLayer(input_size=utils.shape_util.matrix_len(*input_shape), output_size=2, activation="softmax")
    probs = dense.forward(out)
    loss = cross_entropy(probs, label)
    print(loss)
    print(probs)


IMAGE_PATH_1 = './cnn_pure_python/data/1/Cat-Train (1).jpeg'
# IMAGE_PATH_2 = './cnn_pure_python/data/image-103.jpg'

img_matrix1 = utils.image_util.convert_to_list(IMAGE_PATH_1, size=120)
# img_matrix2 = utils.image_util.convert_to_list(IMAGE_PATH_2, size=120)
forward(img_matrix1, label=0)
# forward(img_matrix2, label=1)
