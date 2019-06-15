import numpy as np
import scipy.special as scsp


class NeuronalesNetz:

    def __init__(self, in_nodes, out_nodes, hid_nodes, hid_layers, lr, activation_func):
        # Initialising all the input parameters
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.hid_nodes = hid_nodes

        # Hidden layer parameter must be decremented by one
        self.hid_layers = hid_layers - 1
        self.learn_rate = lr

        # Initialising all the weight matrices with values in a probability distribution
        # around 0 and 1 / sqrt(number of incoming links)
        # First initialize weights between input and hidden layer(s)
        self.weight_input_hidden = np.random.normal(0.0, pow(self.in_nodes, -0.5), (self.hid_nodes, self.in_nodes))

        # Then the weights between the hidden layers
        self.weight_hidden = np.asarray(
            [np.random.normal(0.0, pow(hid_nodes, -0.5), (self.hid_nodes, hid_nodes)) for _ in range(self.hid_layers)])

        # And last but not least the weights between the last hidden layer and the output layer
        self.weight_hidden_output = np.random.normal(0.0, pow(self.out_nodes, -0.5), (self.out_nodes, self.hid_nodes))

        # Initialisation of the activation function and its derivative
        if activation_func == "sigmoid":
            self.activation_func = lambda x: scsp.expit(x)
            self.derivative_func = lambda x: x * (1.0 - x)
        elif activation_func == "tanh":
            self.activation_func = lambda x: np.tanh(x)
            self.derivative_func = lambda x: 1 - x ** 2
        elif activation_func == "ReLU":
            self.activation_func = lambda x: np.maximum(x, 0)
            self.derivative_func = lambda x: np.heaviside(x, 0)
        else:
            raise ValueError("No such activation function")
        pass

    def forward_propagation(self, x):
        x = np.array(x, ndmin=2).T

        # Forward propagation of the input signal
        val_in = self.activation_func(np.dot(self.weight_input_hidden, x))
        val_hid = val_in
        for index in range(0, self.hid_layers):
            val_hid = self.activation_func(np.dot(self.weight_hidden[index], val_hid))
        val_out = self.activation_func(np.dot(self.weight_hidden_output, val_hid))

        return val_out

    def train(self, x, t):
        x = np.array(x, ndmin=2).T
        t = np.array(t, ndmin=2).T

        # First forward propagate the signal to be able to calculate the errors
        hidden_vals = []

        val_in = self.activation_func(np.dot(self.weight_input_hidden, x))
        val_hid = val_in
        for index in range(0, self.hid_layers):
            val_hid = self.activation_func(np.dot(self.weight_hidden[index], val_hid))
            hidden_vals.append(val_hid)
        val_out = self.activation_func(np.dot(self.weight_hidden_output, val_hid))

        # Local variable for reverse iterations
        reverse = self.hid_layers - 1

        # Calculating the error for each layer
        # First calculate error for last layer and last hidden layer
        errors = [t - val_out, np.dot(np.transpose(self.weight_hidden_output), t - val_out)]
        # Then calculate the error for the rest of the layers (if any)
        for index in range(0, self.hid_layers):
            error = np.dot(np.transpose(self.weight_hidden[reverse - index]), errors[index + 1])
            errors.append(error)

        # Now backpropagate the errors using Gradient descent, each weight matrix will be updated
        # First adjust weights for last layer. Note: if there aren't more than two (one) hidden layer, we must
        # use the output of the first (input) layer
        self.weight_hidden_output += self.learn_rate * np.dot(errors[0] * self.derivative_func(val_out), np.transpose(
            hidden_vals[reverse]) if self.hid_layers > 0 else np.transpose(val_in))

        # Then for all the hidden layers. Note: for the first hidden layer the input is val_in as it's the
        # first layer after the input layer
        for index in range(0, self.hid_layers):
            self.weight_hidden[reverse - index] += self.learn_rate * np.dot(
                errors[index + 1] * self.derivative_func(hidden_vals[reverse - index]),
                np.transpose(hidden_vals[reverse - index - 1]) if index < reverse else np.transpose(val_in))

        # And then for the first layer
        self.weight_input_hidden += self.learn_rate * np.dot(errors[len(errors) - 1] * self.derivative_func(val_in),
                                                             np.transpose(x))
        pass
