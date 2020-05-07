import mnist
import time
import numpy as np
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt

"""
__author__ = "Bastian Kersting"
__version__ = "1.2"
"""


# Adding noise to an image. Either Gauss noise or salt and pepper
def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out


def test_network(layers, lr, train_images, train_labels, test_images, test_labels, iterations, is_noisy):
    start_time = time.time()
    overll_performance = []
    round = 1

    for _ in range(iterations):

        input_nodes = 784
        output_nodes = 10
        # 2/3 * input_nodes?
        hidden_nodes = 517
        hidden_layers = layers
        lr = lr
        activation_function = "sigmoid"

        network = NeuralNetwork(input_nodes, output_nodes, hidden_nodes, hidden_layers, lr,
                                                activation_function)

        # Training
        for index in range(len(train_images)):
            current_target = np.zeros(10) + 0.1
            current_target[train_labels[index]] = 0.99
            current_input = []
            for index_w in range(28):
                for index_h in range(28):
                    current_input.append((train_images[0:60000][index][index_w][index_h] / 255.0 * 0.99) + 0.01)
            network.train(current_input, current_target)

        # Testing
        scorecard = []
        for index in range(len(test_images)):
            current_test = []
            for index_w in range(28):
                for index_h in range(28):
                    current_test.append((test_images[0:10000][index][index_w][index_h] / 255.0 * 0.99) + 0.01)

            if is_noisy:
                current_test = noisy("s&p", np.asarray(current_test))
            res = network.forward_propagation(current_test)

            if np.argmax(res) == test_labels[index]:
                scorecard.append(1.0)
            else:
                scorecard.append(0.0)

        # Calculating performance
        performance = float(np.asarray(scorecard).sum()) / float(len(scorecard))
        overll_performance.append(performance)
        print(round, " round(s) finished")
        print("This rounds performance= ", performance * 100)
        round += 1

    end_time = (time.time() - start_time) / 60

    the_real_performance = (np.asarray(overll_performance).sum() / float(len(overll_performance))) * 100
    return end_time, the_real_performance


train_im = mnist.train_images()
train_lbs = mnist.train_labels()
test_ima = mnist.test_images()
test_lbs = mnist.test_labels()
iters = 1
with_noise = True

# Comparing two networks, one with one layer, the other one with five
multilayered_res = []
multilayered_time = 0
for _ in range(5):
    elapsed_time, performance = test_network(5, 0.0125, train_im, train_lbs, test_ima, test_lbs, iters, with_noise)
    multilayered_res.append(performance)
    multilayered_time += elapsed_time

singlelayered_res = []
singlelayered_time = 0
for _ in range(5):
    elapsed_time, performance = test_network(1, 0.22, train_im, train_lbs, test_ima, test_lbs, iters, with_noise)
    singlelayered_res.append(performance)
    singlelayered_time += elapsed_time

# Plotting the results
fig, ax = plt.subplot()

y_val = range(5)
ax[0].plot(multilayered_res, y_val, 'go', label="Neural Network with five layers", color="blue")
ax[1].plot(singlelayered_res, y_val, 'go', label="Neural Network with one layer", color="green")
plt.show()

# print("Elapsed minutes: ", elapsed_time)
# print("My network's performance (%): ", performance)
