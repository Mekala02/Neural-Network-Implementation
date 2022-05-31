import numpy as np
import pickle

class network():

    def __init__(self):
        # Weights: list of numpy arrays .
        # Biases: list of numpy arrays .
        # Activations: list of activation functions.
        self.weights, self.biases, self.activations = [] ,[] , []
        # Layer count
        self.L = 0

    # Calculates element vise sigmoid function.
    @staticmethod
    def Sigmoid(z, derivative=0):
        # Normal sigmoid function
        sigmoid = 1 / (1 + np.exp(-z))
        # Calculates and returns derivitive of Sigmoid function.
        if derivative:
            return sigmoid * (1 - sigmoid)
        return sigmoid

    # Calculates element vise ReLu function.
    @staticmethod
    def ReLu(z, derivative=0):
        if derivative:
            return 1 * (z > 0)
        return z * (z >= 0)

    # Calculates element vise Softmax function.
    @staticmethod
    def Softmax(x, derivative=0):
        exponent = np.exp(x - np.max(x))
        means = exponent.sum(axis=0)
        softmax = exponent / means
        if derivative:
            return exponent / means * ((1 - np.identity(len(x))) @ exponent)
        return softmax

    def add_layer(self, node_count, activation, input_len=0, kernel_initializer=None):
        if input_len:
            self.previous_node_count = input_len
        # Assigning weights using standart distirbution
        layer = np.random.randn(node_count, self.previous_node_count)
        # If we want He Initialization multiplying by math.sqrt(2./len(self.weights)-1) term.
        if kernel_initializer == "He":
            layer *= np.sqrt(2. / self.previous_node_count)
        # If we want He Initialization multiplying by math.sqrt(1./len(self.weights)-1) term.
        elif kernel_initializer == "Xavier":
            layer *= np.sqrt(1. / self.previous_node_count)
        self.weights.append(layer)
        # All biases 0.
        self.biases.append(np.zeros((layer.shape[0], 1)))
        # Keeping track of layers activation function.
        self.activations.append(activation)
        # Keeping track of layers node count becouse if we want to implement new layer we will
        # use this value.
        self.previous_node_count = node_count
        self.L += 1
    
    def unwrap(self, onehot_form):
        classes = []
        for column in np.hsplit(onehot_form, onehot_form.shape[1]):
            index = np.where(column == column.max())[0][0]
            classes.append(index)
        return np.array(classes)

    def predict(self, input_data, one_hot=0):
        # Our activaion functions result, Initialy its input data.
        result = input_data
        # Iterating over all layers and calulating each of them according to previus ones.
        for weight_vector, bias_vector, activation in zip(self.weights, self.biases, self.activations):
            result = activation(weight_vector @ result + bias_vector)
        if one_hot:
            return result
        return self.unwrap(result)

    def save_network(self, file_name):
        # For saving the trained model.
        file = open(file_name+'.txt','wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load_network(self, file_name):
        # For loading the trained model.
        """try load self.name.txt"""
        file = open(file_name+'.txt','rb')
        dataPickle = file.read()
        file.close()
        self.__dict__ = pickle.loads(dataPickle)