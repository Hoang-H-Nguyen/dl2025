import math
import matplotlib.pyplot as plt
from utils.conv_layer import ConvLayer
import utils.image_util
import numpy as np

IMAGE_PATH = './cnn_pure_python/data/image-66.jpg'

img_matrix = utils.image_util.convert_to_list(IMAGE_PATH, size=120)
print(np.array(img_matrix).shape)
conv = ConvLayer(filter_size= 32, num_filters=4)
out = conv.forward(img_matrix)
print(np.array(out).shape)
plt.imshow(out)
plt.show()

# image = [
#   [[1,2,3],    [4,5,6],    [7,8,9],    [10,11,12], [13,14,15]],
#   [[16,17,18], [19,20,21], [22,23,24], [25,26,27], [28,29,30]],
#   [[31,32,33], [34,35,36], [37,38,39], [40,41,42], [43,44,45]],
#   [[46,47,48], [49,50,51], [52,53,54], [55,56,57], [58,59,60]],
#   [[61,62,63], [64,65,66], [67,68,69], [70,71,72], [73,74,75]],
# ]

# conv = ConvLayer(filter_size=3, num_filters=4)
# out = conv.forward(image)
# print(np.array(out).shape)

