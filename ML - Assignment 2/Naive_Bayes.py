#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np

# Load data
data = []
with open("./Assignment2/naive_bayes_data.txt","r") as f:
    data = f.readlines()

# Clean data
cleaned = []
for doc in data:
    temp = {}
    words = doc.split()
    temp["category"] = words[0]
    temp["sentiment"] = words[1]
    temp["doc_id"] = words[2]
    temp["doc"] = " ".join(words[3:])
    cleaned.append(temp)

# Split data into training and validation data
train = cleaned[:int(0.8*len(cleaned))]
validation = cleaned[int(0.8*len(cleaned)):]

# Calculate priors
neg_count = 0
pos_count = 0
for doc in cleaned:
    if(doc["sentiment"] == "neg"):
        neg_count+=1
    elif(doc["sentiment"] == "pos"):
        pos_count+=1

neg = float(neg_count)/len(cleaned)
pos = float(pos_count)/len(cleaned)

# Get all the unique words
unique = set()
for doc in cleaned:
    for word in doc["doc"].split(" "):
        unique.add(word)
# print(len(unique))

# Get a list of all the positive and negative words
all_words_pos = []
all_words_neg = []
for doc in cleaned:
    if(doc["sentiment"]=="pos"):
        for word in doc["doc"].split(" "):
            all_words_pos.append(word)
    elif(doc["sentiment"]=="neg"):
        for word in doc["doc"].split(" "):
            all_words_neg.append(word)

# Get counts of all the words
count_words_pos = {}
count_words_neg = {}

for word in unique:
    count_words_neg[word] = all_words_neg.count(word)
    count_words_pos[word] = all_words_pos.count(word)

# Load from pickle to save time
with open("count_pos_train.pkl","rb") as f:
    count_words_pos_train = pickle.load(f)

with open("count_neg_train.pkl","rb") as f:
    count_words_neg_train = pickle.load(f)

# Calculate the Accuracy by predicting the sentiment of the validation data
correct = 0
wrong = 0
# Predicted and Actual
pp = 0
nn = 0
pn = 0
fn = 0
for test in validation:
    p = pos
    for word in test["doc"].split(" "):
        try:
            cond = (count_words_pos_train[word])
        except Exception as e:
            cond = 1
            print("This is a new word")
        p *= float(cond+1)/(len(all_words_pos)+len(unique))

    n = neg
    for word in test["doc"].split(" "):
        try:
            cond = (count_words_neg_train[word])
        except Exception as e:
            cond = 1
            print("This is a new word")
        n *= float(cond+1)/(len(all_words_neg)+len(unique))

    res  = "neg"
    if(p>n): res = "pos"
    if(res==test["sentiment"]): correct = correct + 1
    else: wrong = wrong +1

    # Calculate the true positive,true negative, false positive and false negative
    if(res==test["sentiment"] and test["sentiment"]=="pos"): pp = pp + 1
    if(res==test["sentiment"] and test["sentiment"]=="neg"): nn = nn + 1
    if(res!=test["sentiment"] and test["sentiment"]=="neg"): pn = pn + 1 #FP
    if(res!=test["sentiment"] and test["sentiment"]=="pos"): fn = fn + 1 #FN

acc = correct/float(wrong+correct)
print(acc)

precision = pp/float(pp+pn)
print(precision)

recall = pp/float(pp+np)
print(recall)

f1 = 2*((precision*recall)/(precision+recall))
print(f1)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    get_ipython().magic(u'matplotlib inline')

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

# Plot the confusion matrix
plot_confusion_matrix(cm           = np.array([[ 545,  608],
                                              [  146,  1084]]),
                      normalize    = False,
                      target_names = ['Pos', 'Neg'],
                      title        = "Confusion Matrix")
