from typing import List
import numpy as np
from typing import List, Literal


def sigmoid(z):
    """
    z is assumed to be vector and then the sigmoid is applied element wise
    """
    return 1.0/(1.0+np.exp(-z))


def perceptron(z):
    z[z > 0] = 1.0
    z[z < 0] = 0.0
    return z


class NeuralNet(object):
    def __init__(self, size: List, neuron_type: Literal = "sigmoid"):
        """
        ## Size
        Size represent number of neuron in neural network layers.
        For example : [I,n1,n2,n3....nm] : means neural network with I neurons in Input layer
        n2 in second hidden layer  ... nk in the kth layer and so on 


        n1 : It is the first hidden layer

        Thus total number of hidden layers are  = len(sizes) -2
        Total layers are = len(size)-1

        ## neuron_type
        "sigmoid" or "perceptron"

        ## Note
        The first index must contain the number of input neurons
        """
        self.num_hidden_layers = len(size)-2
        self.size = size
        self.biases = [np.random.rand(i, 1) for i in self.size[1:]]
        self.weights = [np.random.rand(i, i-1) for i in self.size[1:]]

        self.neuron_type = neuron_type
        self.activation_func = {"sigmoid": sigmoid,
                                "perceptron": perceptron}[neuron_type]

    def describe(self):

        print(f"Neuron Type = {self.neuron_type}")
        print(f"Number of hidden layers = {self.num_hidden_layers}")

    def feedforward(self, a: np.ndarray):
        """
        ## About
        Given input a return output of the neural net on the given input vetor
        a is assumed to be (n,1) vector 
        """
        a = a.reshape((-1, 1))
        for w, b in zip(self.weights, self.biases):
            a = self.activation_func(w@a + b)
        return a

    def fit(self, train_data: np.ndarray, epochs: int, mini_batch_size, eta):

        n = train_data.shape[0]
        for j in range(epochs):
            np.random.shuffle(train_data)
            mini_batches = [train_data[i:i+mini_batch_size]
                            for i in range(0, n, mini_batch_size)]
            for batch in mini_batches:
                self.fit_mini_batch(batch, eta)

    def fit_mini_batch(self, mini_batch, eta):
        """
        The `mini_batch` is a list of tuples `(x, y)`, and `eta` is the learning rate
        """
        n = len(mini_batch)
        grad_weights = [np.zeros_like(w) for w in self.weights]
        grad_biases = [np.zeros_like(b) for b in self.biases]

        for x, y in mini_batch:
            delta_grad_w, delta_grad_b = self.backprop(x, y)
            grad_weights = [w+dw for w, dw in zip(self.weights, delta_grad_w)]
            grad_biases = [b+db for b, db in zip(self.biases, delta_grad_b)]

        self.w = [w - (eta/(n))*dw for w,
                  dw in zip(self.weights, grad_weights)]
        self.b = [b - (eta/(n))*db for b,
                  db in zip(self.biases, grad_biases)]

    def backprop(self, x, y):
        pass

    def score(self, test_data):
        """ Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural 
        network’s output is assumed to be the index of whichever 
        neuron in the final layer has the highest activation.
        """
        scores = [(np.argmax(self.feedforward(x), y))for x, y in test_data]

        return np.sum(int(x == y) for x, y in scores)
