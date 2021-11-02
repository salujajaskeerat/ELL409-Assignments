from typing import List
import numpy as np
from typing import List, Literal


def sigmoid(z):
    """
    z is assumed to be vector and then the sigmoid is applied element wise
    """
    return 1.0/(1.0+np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z)*(1.0-sigmoid(z))


def perceptron(z):
    z[z > 0] = 1.0
    z[z < 0] = 0.0
    return z


class NeuralNet(object):
    def __init__(self, size: List, neuron_type: Literal = "sigmoid"):
        """
        # Size
        Size represent number of neuron in neural network layers.
        For example : [I,n1,n2,n3....nm] : means neural network with I neurons in Input layer
        n2 in second hidden layer  ... nk in the kth layer and so on


        n1 : It is the first hidden layer

        Thus total number of hidden layers are  = len(sizes) -2
        Total layers are = len(size)-1

        # neuron_type
        "sigmoid" or "perceptron"

        # Note
        The first index must contain the number of input neurons
        """
        self.num_layers = len(size)
        self.size = size
        self.biases = [np.random.randn(i, 1) for i in self.size[1:]]
        self.weights = [np.random.randn(i, j) for i, j in zip(
            self.size[1:], self.size[:-1])]

        self.neuron_type = neuron_type
        self.activation_func = {"sigmoid": sigmoid,
                                "perceptron": perceptron}[neuron_type]

    def describe(self):

        print(f"Neuron Type = {self.neuron_type}")
        print(f"Number of hidden layers = {self.num_hidden_layers}")

    def feedforward(self, a: np.ndarray):
        """
        # About
        Given input a return output of the neural net on the given input vetor
        a is assumed to be (n,1) vector
        """
        a = a.reshape((-1, 1))
        for w, b in zip(self.weights, self.biases):
            a = self.activation_func(np.dot(w, a) + b)
        return a

    def fit(self, train_data: np.ndarray, epochs: int, mini_batch_size, eta, test_data=None):

        n = len(train_data)
        for j in range(epochs):
            np.random.shuffle(train_data)
            mini_batches = [train_data[i:i+mini_batch_size]
                            for i in range(0, n, mini_batch_size)]
            for batch in mini_batches:
                self.fit_mini_batch(batch, eta)
            if(test_data):
                print(
                    f"Epoch:{j+1} Test data score : ={self.score(test_data)} / {len(test_data)}")

    def fit_mini_batch(self, mini_batch, eta: float):
        """
        The `mini_batch` is a list of tuples `(x, y)`, and `eta` is the learning rate
        """
        n = len(mini_batch)
        net_grad_weights = [np.zeros_like(w) for w in self.weights]
        net_grad_biases = [np.zeros_like(b) for b in self.biases]

        for x, y in mini_batch:
            delta_grad_w, delta_grad_b = self.backprop(x, y)
            net_grad_weights = [w+dw for w,
                                dw in zip(net_grad_weights, delta_grad_w)]
            net_grad_biases = [b+db for b,
                               db in zip(net_grad_biases, delta_grad_b)]

        self.weights = [w - (eta/(n))*dw for w,
                        dw in zip(self.weights, net_grad_weights)]
        self.biases = [b - (eta/(n))*db for b,
                       db in zip(self.biases, net_grad_biases)]

    def cost_derivative(self, a_out, y):
        # w.r.t to the sigmoid neuron
        return (a_out-y)

    def backprop(self, x, y):
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        zs = []

        # Forwards propagations
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_derivative(zs[-1])
        db[-1] = delta
        dw[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            db[-l] = delta
            dw[-l] = np.dot(delta, activations[-l-1].transpose())

        return (dw, db)

    def score(self, test_data):
        """ Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural 
        network’s output is assumed to be the index of whichever 
        neuron in the final layer has the highest activation.
        """
        scores = [(np.argmax(self.feedforward(x)), y)for x, y in test_data]

        return np.sum(int(x == y) for x, y in scores)
