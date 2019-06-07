"""
The main home made Neural Network implementation.

Written by Thomas Holzheu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class NeuralNet:
    """End-to-end customizable Neural Network implementation with numpy."""
    def __init__(self, input_size, topology):
        """
        Specify the Network's architecture.

        Initializes weights and biases (He initialization) of the Network.

        Parameters
        ----------
        input_size (int) : Size of the input layer of the Network
        topology (tuple) : Tuple that contains the number of nodes per layer
                           (e.g. (256, 128) --> 2 layers of 256 and 128 nodes respectively)
        """
        self.input_size = input_size
        self.topology = topology
        self.weights = []
        self.biases = []
        prev_layer_size = input_size
        for layer_size in topology:
            self.weights.append(np.random.randn(layer_size, prev_layer_size) * np.sqrt(2 / prev_layer_size))
            self.biases.append(np.zeros(layer_size))
            prev_layer_size = layer_size
    def softmax(self, vec):
        """Calculate the softmax of a vector."""
        ex_vec = np.exp(vec)
        return (ex_vec / sum(ex_vec))
    def forwardprop(self, x, y):
        """
        Forward propagation of the Network.

        Parameters
        ----------
        x (np array) : Current training example
        y (np array) : Current label example

        Returns
        -------
        ops (list) : Contains every vector calculated during the forward propagation from input to output (both included)
        loss (int) : Loss of the given the current training example
        """
        ops = [x]
        for i in range(len(self.topology)):
            z = np.matmul(self.weights[i], ops[-1]) + self.biases[i]
            if (i == len(self.topology) - 1):
                a = self.softmax(z)
            else:
                a = np.where(z < 0, 0, z)
            ops.append(z)
            ops.append(a)
        loss = -1 * np.log(np.dot(ops[-1], y))
        return (ops, loss)
    def backprop(self, l_rate, ops, y):
        """
        Backpropagation of the Network.

        Parameters
        ----------
        l_rate (float)
        ops (list) : Contains every vector calculated during the forward propagation from input to output (both included)
        y (np array) : Current label example

        Returns
        -------
        List containing every gradient calculated during the backpropagation
        """
        grads = []
        grad_z = ops.pop() - y
        ops.pop()
        for i in range(len(self.topology)):
            grads.append(np.outer(grad_z, ops.pop()))
            grads.append(grad_z)
            if (i < len(self.topology) - 1):
                grad_prev_a = np.dot(self.weights[-i - 1].T, grad_z)
                grad_actv_funct = np.where(ops.pop() > 0, 1, 0)
                grad_z = grad_actv_funct * grad_prev_a

        # update weights and biases
        ret_grads = grads.copy()
        for i in range(len(self.topology)):
            a = grads.pop()
            b = grads.pop()
            self.biases[i] -= l_rate * a
            self.weights[i] -= l_rate * b
        return (ret_grads)
    def plot_stats(self, stats):
        """Plots training and validation's loss and accuracy."""
        df = pd.DataFrame(stats)
        df_loss = df.iloc[:, :2]
        df_acc = df.iloc[:, 2:4]
        f, ax = plt.subplots(figsize=(20, 15))
        plt.plot(df_loss, linewidth=4)
        f.legend(("Training", "Validation"), fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel('Steps', fontsize=25)
        plt.ylabel("Loss", fontsize=25)
        plt.title("Cross Entropy over whole dataset", fontsize=40);

        f, ax = plt.subplots(figsize=(20, 15))
        plt.plot(df_acc, linewidth=4)
        f.legend(("Training", "Validation"), fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlabel('Steps', fontsize=25)
        plt.ylabel("Accuracy", fontsize=25)
        plt.title("Accuracy over whole dataset", fontsize=40);
    def update_stats(self, t_features, t_labels, v_features, v_labels):
        """Saves new stats using the current weights and bias."""
        t_loss = 0
        t_acc = 0
        v_loss = 0
        v_acc = 0
        for x, y in zip(t_features, t_labels):
            ops, loss = self.forwardprop(x, y)
            t_loss += loss
            ind = np.argmax(ops[-1])
            t_acc += y[ind]
        t_loss /= len(t_features)
        t_acc /= len(t_features)
        for x, y, in zip(v_features, v_labels):
            ops, loss = self.forwardprop(x, y)
            v_loss += loss
            ind = np.argmax(ops[-1])
            v_acc += y[ind]
        v_loss /= len(v_features)
        v_acc /= len(v_features)
        print('training:\n loss:', t_loss, 'accuracy:', t_acc)
        print('validation:\n loss:', v_loss, 'accuracy', t_acc)
        print()
        return ((t_loss, v_loss, t_acc, v_acc))
    def next_batch(self, features, labels, batch_size):
        """Yield the next batch for training."""
        for i in range(0, len(features), batch_size):
            yield features[:][i:i + batch_size], labels[i:i + batch_size]
    def training(self, t_features, t_labels, v_features, v_labels, l_rate, epochs, batch_size, plot_steps):
        """
        Function to train the Network.

        Parameters
        ----------
        t_features (np array) : Training features
        t_labels (np array) : Training labels
        v_features (np array) : Validation features
        v_labels (np array) : Validation labels
        l_rate (float)
        epochs (int)
        batch_size (int)
        plot_steps (int) : The amount of steps to take before plotting

        Returns
        -------
        List of tuples containing training and validation's loss and accuracy
        """
        steps = 0
        stats = []
        for epoch_nb in range(epochs):
            generator = self.next_batch(t_features, t_labels, batch_size)
            for batch_x, batch_y in generator:
                batch_loss = 0
                for x, y in zip(batch_x, batch_y):
                    ops, loss = self.forwardprop(x, y)
                    batch_loss += loss
                    if (steps % plot_steps == 0):
                        stats.append(self.update_stats(t_features, t_labels, v_features, v_labels))
                    steps += 1
                batch_loss /= batch_size
                self.backprop(l_rate, ops, y)
        return (stats)
