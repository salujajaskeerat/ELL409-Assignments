from typing import List
import numpy as np
from typing import List, Literal
from neuron import sigmoid_neuron, relu_neuron, tanh_neuron, soft_plus
from cost_funcs import quadratic_cost, cross_entropy_cost


# import logging
# logger = logging.getLogger('Debug Logger')
# logger.setLevel(logging.DEBUG)

# # create console handler with a higher log level
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# # add the handlers to logger
# logger.addHandler(ch)


# NOTE : Last layer must always be a sigmoid layer

def perceptron(z):
    z[z > 0] = 1.0
    z[z < 0] = 0.0
    return z


class NeuralNet(object):
    def __init__(self, size: List, neuron_type: Literal = "sigmoid", cost="cross_entropy_cost"):
        """
        # Size
        Size represent number of neuron in neural network layers.
        For example : [I,n1,n2,n3....nm] : means neural network with I neurons in Input layer
        n2 in second hidden layer  ... nk in the kth layer and so on

        # Cost
        Type of cost function to be used while training thr neural network
        `cross_entropy_cost`(default) and `quadratic_cost`

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
        self.activation_func = {"sigmoid": sigmoid_neuron.f,
                                "relu": relu_neuron.f,
                                "tanh": tanh_neuron.f,
                                "soft_plus": soft_plus.f,

                                }[neuron_type]
        self.activation_fun_derivative = {"sigmoid": sigmoid_neuron.f_prime,
                                          "relu": relu_neuron.f_prime,
                                          "tanh": tanh_neuron.f_prime,
                                          "soft_plus": soft_plus.f_prime
                                          }[neuron_type]

        self.cost = {"quadratic_cost": quadratic_cost,
                     "cross_entropy_cost": cross_entropy_cost}[cost]

    def describe(self):

        print(f"Neuron Type = {self.neuron_type}")
        print(f"Number of hidden layers = {self.num_hidden_layers}")

    def ith_layer_out(self, a: np.ndarray, k):
        a = a.reshape((-1, 1))
        for w, b in zip(self.weights[:-1*k], self.biases[:-1*k]):
            a = self.activation_func(np.dot(w, a) + b)

        return a

    def feedforward(self, a: np.ndarray):
        """
        # About
        Given input a return output of the neural net on the given input vetor
        a is assumed to be (n,1) vector
        """
        a = a.reshape((-1, 1))
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            a = self.activation_func(np.dot(w, a) + b)

        # Last layer is sigmoid
        a = sigmoid_neuron.f(np.dot(self.weights[-1], a)+self.biases[-1])
        return a

    def fit(self, train_data: np.ndarray, epochs: int, mini_batch_size, eta, test_data=None, lmda=0, decay_rate=2, decay_after_epoch=5, verbose=True):
        """
        ## train_data

        `format`: numpy.array([(x1,y1), (x2,y2)]) where each `xi` is itself a numpy array with dimension `n*1` where n is the 
        number of neurons in first layer and `yi` have dimesnion `m*1` where m is the number of neuron in output layer 
        ## test_data
        if passed as `None` the model is only trained on the training data
        else the scores for test data are also evaluated for each epoch

        ## lmda (regularization cofficient)
        lmda is the regularization cofficient 
        """
        n = len(train_data)
        train_accuracy, train_cost, test_accuracy, test_cost = [], [], [], []

        # print("Scores -->")
        for j in range(epochs):

            # Decay scehdule
            if(j % decay_after_epoch == 0):
                eta = eta/decay_rate

            np.random.shuffle(train_data)
            mini_batches = [train_data[i:i+mini_batch_size]
                            for i in range(0, n, mini_batch_size)]
            for batch in mini_batches:
                self.fit_mini_batch(batch, eta, lmda)

            train_accuracy.append(self.accuracy(train_data, True))
            train_cost.append(self.net_cost(train_data, lmda))

            if(test_data):
                test_accuracy.append(self.accuracy(test_data))
                test_cost.append(self.net_cost(test_data, lmda, True))
            if verbose:
                print(
                    f'Epoch :{j+1} complete: test_accuracy: {test_accuracy[-1]} , train_accuracy :{train_accuracy[-1]}',  flush=True)
        return (train_accuracy, train_cost, test_accuracy, test_cost)

    def fit_mini_batch(self, mini_batch, eta: float, lmda: float):
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

        # Update rule and regularization
        self.weights = [w * (1-eta*(lmda/n)) - (eta/(n))*dw for w,
                        dw in zip(self.weights, net_grad_weights)]
        self.biases = [b*(1-eta*(lmda/n)) - (eta/(n))*db for b,
                       db in zip(self.biases, net_grad_biases)]

    def backprop(self, x, y):
        db = [np.zeros(b.shape) for b in self.biases]
        dw = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        a = x
        activations = [x]
        zs = []

        # Forwards propagations
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, a)+b
            zs.append(z)
            a = self.activation_func(z)
            activations.append(a)

        # Last layer is always sigmoid
        z = np.dot(self.weights[-1], a)+self.biases[-1]
        zs.append(z)
        a = sigmoid_neuron.f(z)
        activations.append(a)

        # backward pass
        delta = (self.cost).delta(
            zs[-1], activations[-1], y, sigmoid_neuron.f_prime)  # last layer is sigmoid so
        db[-1] = delta
        dw[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_fun_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            db[-l] = delta
            dw[-l] = np.dot(delta, activations[-l-1].transpose())

        return (dw, db)

    def net_cost(self, data, lmda, convert_format=False):
        cost = 0
        n = len(data)

        for x, y in data:
            a = self.feedforward(x)
            if(convert_format):
                y = vectorized_result(y)
            cost += (self.cost).cost(a, y)/n
        cost += 0.5*(lmda/n) * \
            sum(np.linalg.norm(w)**2 for w in self.weights)

        return cost

    def predict(self, test_data: np.ndarray):

        scores = [(np.argmax(self.feedforward(x)))for x, y in test_data]
        return scores

    def accuracy(self, data, convert_format=False):
        """ Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network’s output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        if(convert_format):
            scores = [(np.argmax(self.feedforward(x)), np.argmax(y))
                      for x, y in data]

        else:
            scores = [(np.argmax(self.feedforward(x)), y)for x, y in data]

        return (np.sum(int(x == y) for x, y in scores))/len(data)


def vectorized_result(j):

    y = np.zeros((10, 1))
    y[j] = 1.0
    return y
