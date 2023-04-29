#!/usr/bin/env python
# coding: utf-8

import os
import cv2
from random import random,seed,randrange
from csv import reader
from math import exp

path = "./data/train/train"
resized_path = "./data/train/resized"

# Change the images to grayscale and resize
train = []
for f in os.listdir(path):
    img = cv2.imread(os.path.join(path,f),0)
    dim = (50,50)
    try:
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    except Exception as e:
        print(e)
    cv2.imwrite(resized_path +"/" + f,resized)

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = float(row[i]) / 255.0

# Add the label to the end of the one dimensional array
path = "./data/train/resized"
train = []
for f in os.listdir(path):
    temp = {}
    img = cv2.imread(os.path.join(path,f))
    temp["image"] = img[:,:,2].flatten()
    label = (f.split(".")[0])
    if label == "dog":
        temp["label"] = 1
    else:
        temp["label"] = 0
    train.append(temp)

# Normalize the training dataset
normalize_dataset(train)

# Do the above steps for the test/validation data
path = "./data/train/copy"
test = []
for f in os.listdir(path):
    img = cv2.imread(os.path.join(path,f))
    temp = list(img[:,:,2].flatten())
    label = (f.split(".")[0])
    if label == "dog":
        temp.append(1)
    else:
        temp.append(0)
    test.append(temp)

normalize_dataset(test)

# Store the data in a csv file for easy accessiblity
with open('sample3.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(train[:500])

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for image in csv_reader:
            if not image:
                continue
            dataset.append(image)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for image in dataset:
        image[column] = float(image[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [image[column] for image in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for image in dataset:
        image[column] = lookup[image[column]]
    return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage and confusion matrix
def accuracy_metric(actual, predicted):
    correct = 0
    dd = 0
    cc = 0
    dc = 0
    cd = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
        if actual[i] == predicted[i] and actual[i] == 0:
            cc +=1
        if actual[i] == predicted[i] and actual[i] == 1:
            dd +=1
        if actual[i] != predicted[i] and actual[i] == 0:
            cd +=1
        if actual[i] != predicted[i] and actual[i] == 1:
            dc +=1
    print("dd" + str(dd))
    print("cc" + str(cc))
    print("cd" + str(cd))
    print("dc" + str(dc))
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def run(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for image in fold:
            image_copy = list(image)
            test_set.append(image_copy)
            image_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        print(predicted)
        actual = [image[-1] for image in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Calculate neuron activation for an input
def compute_activation(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Forward propagate input to a network output
def feed_forward(network, image):
    inputs = image
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = compute_activation(neuron['weights'], inputs)
            neuron['output'] = 1.0 / (1.0 + exp(-activation))
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['change'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['change'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, image, learning_rate):
    for i in range(len(network)):
        inputs = image[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['change'] * inputs[j]
            neuron['weights'][-1] += learning_rate * neuron['change']

# Train a network for a fixed number of epochs
def train_network(network, train, learning_rate, num_epochs, output_neurons):
    for epoch in range(num_epochs):
        for image in train:
            outputs = feed_forward(network, image)
            expected = [0 for i in range(output_neurons)]
            expected[image[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, image, learning_rate)

# Initialize a network with the input layer, one hidden layer and one output layer
def initialize_network(input_neurons, hidden_neurons, output_neurons):
    network = list()
    hidden_layer = []
    for i in range(hidden_neurons):
        temp = {}
        temp["weights"] = []
        for i in range(input_neurons + 1):
            temp["weights"].append(random())
        hidden_layer.append(temp)
    network.append(hidden_layer)

    output_layer = []
    for i in range(output_neurons):
        temp = {}
        temp["weights"] = []
        for i in range(hidden_neurons + 1):
            temp["weights"].append(random())
        output_layer.append(temp)
    network.append(output_layer)
    return network

# Choose the maximum among the output neurons to be the prediction
def predict(network, image):
    outputs = feed_forward(network, image)
    return outputs.index(max(outputs))

# Backpropagation Algorithm With Gradient Descent
def back_propagation(train, test, learning_rate, num_epochs, hidden_neurons):
    input_neurons = len(train[0]) - 1 # This is 2500
    output_neurons = 2
    network = initialize_network(input_neurons, hidden_neurons, output_neurons)
    train_network(network, train, learning_rate, num_epochs, output_neurons)
    predictions = list()
    for image in test:
        prediction = predict(network, image)
        predictions.append(prediction)
    return(predictions)

# load and prepare data
# Convert the csv data in a numpy array of floats for the pixel intensities and integers for the class labels
filename = 'sample2.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
str_column_to_int(dataset, len(dataset[0])-1)

# normalize input variables
normalize_dataset(dataset)

# Run the algorithm
n_folds = 5
learning_rate = 0.5
num_epochs = 300
hidden_neurons = 16
scores = run(dataset, back_propagation, n_folds, learning_rate, num_epochs, hidden_neurons)
print('Accuracy: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
