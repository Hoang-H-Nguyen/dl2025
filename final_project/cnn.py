import math
import matplotlib.pyplot as plt
from utils.conv_layer import ConvLayer
from utils.pooling_layer import MaxPoolingLayer
from utils.dense_layer import DenseLayer
import utils.shape_util
import utils.image_util
from typing import List, Tuple, Dict

class MyCNN:
    def __init__(self, file_config: str):
        self.file_config = file_config
        self.config = self._load_config_from_file(config_file = self.file_config)

        self.input_shape = (self.config["h"], self.config["w"], self.config["c"]) 
        self.num_classes = self.config["num_classes"]

        self.conv = ConvLayer(filter_size=self.config["filter_size"], num_filters=self.config["num_filters"])
        self.max_pool = MaxPoolingLayer(pooling_size=self.config["pooling_size"])

        pooled_h = self.input_shape[0] // 2
        pooled_w = self.input_shape[1] // 2
        pooled_c = 4  # num_filters from conv layer
        dense1_input_size = pooled_h * pooled_w * pooled_c

        self.dense1 = DenseLayer(input_size=dense1_input_size, output_size=self.config["output_size_dense1"], activation="relu")
        self.dense2 = DenseLayer(input_size=self.config["output_size_dense1"], output_size=self.num_classes, activation="softmax")

        # Training history
        self.training_history = {
            'epochs': [],
            'losses': [],
            'accuracies': [],
        }

    def _load_config_from_file(self, config_file: str) -> Dict:
        config = {}        
        try:
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # Parse different value types
                            if key in ['h', 'w', 'c', 'filter_size', 'num_filters', 'pooling_size', 
                                     'output_size_dense1', 'output_size_dense2', 'num_classes']:
                                config[key] = int(value)
                            elif key in ['learning_rate']:
                                config[key] = float(value)
                            else:
                                config[key] = value
                                
        except FileNotFoundError:
            print(f"FileNotFoundError. Using default values.")
            config = self._get_default_config()
        except Exception as e:
            print(f"{e}. Using default values.")
            config = self._get_default_config()
        return config
    
    def _get_default_config(self) -> Dict:
        return {
            'h': 120,
            'w': 120,
            'c': 3,
            'filter_size': 3,
            'num_filters': 4,
            'pooling_size': 2,
            'output_size_dense1': 12,
            'output_size_dense2': 2,
            'num_classes': 3,
            'learning_rate': 0.00001
        }
    
    def _cross_entropy(self, predicted_probs: List, true_label: int) -> float:
        """Compute cross-entropy loss"""
        return -math.log(predicted_probs[true_label] + 1e-15)
    
    def _cross_entropy_derivative(self, predicted_probs: List, true_label: int) -> List:
        """Compute derivative of cross-entropy loss"""
        d_loss = [0.0] * len(predicted_probs)
        d_loss[true_label] = -1.0 / (predicted_probs[true_label] + 1e-15)
        return d_loss
    
    def _argmax(self, probs: List) -> int:
        """Find index of maximum probability"""
        max_idx = 0
        max_val = probs[0]
        for i in range(1, len(probs)):
            if probs[i] > max_val:
                max_val = probs[i]
                max_idx = i
        return max_idx

    def forward(self, input_data: List) -> List:
        x = self.conv.forward(input_data)
        x = self.max_pool.forward(x)

        input_shape1 = utils.shape_util.shape(x)
        input_size1 = utils.shape_util.matrix_len(*input_shape1)
        self.dense1.input_size = input_size1
        x = self.dense1.forward(x)

        input_shape2 = utils.shape_util.shape(x)
        input_size2= utils.shape_util.matrix_len(*input_shape2)
        self.dense2.input_size = input_size2
        x = self.dense2.forward(x)

        return x
    
    def backward(self, predicted_probs: List, true_label: int, learning_rate: float = 0.01):
        # Compute loss gradient
        d_loss = self._cross_entropy_derivative(predicted_probs, true_label)
        
        # Backpropagate through layers
        d_dense1_out = self.dense2.backward(d_loss, learning_rate)
        d_pool_out = self.dense1.backward(d_dense1_out, learning_rate)
        d_conv_out = self.max_pool.backward(d_pool_out)
        d_input = self.conv.backward(d_conv_out, learning_rate)
        
        return d_input
    
    def train_step(self, input_data: List, true_label: int, learning_rate: float = 0.01) -> float:
        """Single training step"""
        predicted_probs = self.forward(input_data)
        # print("predicted_probs in train_step", predicted_probs)
        # print("true_label in train_step", true_label)
        loss = self._cross_entropy(predicted_probs, true_label)
        self.backward(predicted_probs, true_label, learning_rate)
        
        return loss
    
    def train(self, train_data: List[Tuple[List, int]], 
              learning_rate: float = 0.0001,
              epochs: int = 10,
              batch_size: int = 1,
              verbose: bool = True):
        """
        train_data: List of (image_matrix, label) tuples
        """
        if verbose:
            print(f"Starting training for {epochs} epochs...")
            print(f"Training samples: {len(train_data)}")

        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0

            for i, (image_matrix, label) in enumerate(train_data):
                # Adjust label to be 0-indexed if needed
                adjusted_label = label - 1 if label > 0 else label

                # print("Image", i, "Label", adjusted_label)
                loss = self.train_step(image_matrix, adjusted_label, learning_rate)
                total_loss += loss

                pred_class, _ = self.predict(image_matrix)
                if pred_class == adjusted_label:
                    correct_predictions += 1

                if verbose:
                    print(f"  Epoch {epoch+1}/{epochs} - Step {i+1}/{len(train_data)} - Loss: {loss:.4f}")

            # Calculate training metrics
            avg_loss = total_loss / len(train_data)
            accuracy = correct_predictions / len(train_data)
            
            self.training_history['epochs'].append(epoch)
            self.training_history['losses'].append(avg_loss)
            self.training_history['accuracies'].append(accuracy)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}", end="")
        
        if verbose:
            print("\nTraining completed!")

        return self.training_history
    
    def predict(self, input_data: List) -> Tuple[int, List]:
        """
        Make prediction on single input
        
        Returns:
            (predicted_class, probabilities)
        """
        probs = self.forward(input_data)
        predicted_class = self._argmax(probs)
        return predicted_class, probs
    
    def get_model_summary(self) -> str:
        summary = []
        summary.append("CNN Model Summary:")
        summary.append("=============================")
        summary.append(f"Input Shape: {self.input_shape}")
        summary.append(f"Conv Layer: {self.conv.num_filters} filters, size {self.conv.filter_size}x{self.conv.filter_size}")
        summary.append(f"MaxPool Layer: {self.max_pool.pooling_size}x{self.max_pool.pooling_size}")
        summary.append(f"Dense Layer 1: {self.dense1.input_size} -> {self.dense1.output_size} (ReLU)")
        summary.append(f"Dense Layer 2: {self.dense2.input_size} -> {self.dense2.output_size} (Softmax)")
        summary.append(f"Output Classes: {self.num_classes}")
        summary.append("=============================")
        return "\n".join(summary)

def load_data():
    data_list = []
    for label in range(1, 4):
        for img_idx in range(1, 4):
            IMAGE_PATH = f'./final_project/data/{label}/{img_idx}.jpeg'
            img_matrix = utils.image_util.convert_to_list(IMAGE_PATH=IMAGE_PATH, size=120)
            data_list.append( (img_matrix, label) )
    return data_list

def train_cnn_example():
    data = load_data()

    file_config = "./file_config.txt"
    cnn = MyCNN(file_config=file_config)

    print(cnn.get_model_summary())

    history = cnn.train(
        train_data=data,
        epochs=10,
        learning_rate=0.00001,
        verbose=True
    )

    return cnn, history

def plot_training_history(training_history: Dict, save_path: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = training_history['epochs']
    losses = training_history['losses']
    accuracies = training_history['accuracies']

    # Plot losses
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    valid_losses = [loss if not math.isnan(loss) else None for loss in losses]
    ax1.plot(epochs, valid_losses, 'b-o', label='Loss')
        
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.plot(epochs, accuracies, 'g-o', label='Accuracy')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

if __name__ == "__main__":
    model, training_history = train_cnn_example()
    plot_training_history(training_history=training_history, save_path="./final_project/training_history.png")