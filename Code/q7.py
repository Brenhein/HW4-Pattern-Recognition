from operator import itemgetter
from sklearn.metrics import confusion_matrix
import numpy as np
import pprint as pp
import scipy.spatial.distance as sp
import matplotlib.pyplot as plt


def plot_class(ks, ws):
    """Plots the classification effectiveness as a function of k"""
    fig, ax = plt.subplots()
    ax.set_xlabel("K")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("K-NN Classification Accuracy")
    ax.scatter(ks, ws)
    


def classify(k, distances):
    """Using K-NN, classifies the testing points USING the training points"""
    distances_k = sorted(distances, key=itemgetter(0))[:k]
    counts = [0, 0, 0]
    for d in distances_k:
        counts[d[1]] += 1
    return max(enumerate(counts), key=itemgetter(1))[0]+1


def knn(k, train, test):
    """Uses K-NN to classify each of the test points"""
    predicted, actual = [], []
    for test_w, test_patterns in enumerate(test):
        for test_x in test_patterns:
            distances = []
            for train_w, train_patterns in enumerate(train):
                for train_x in train_patterns:
                    distances.append((sp.euclidean(test_x, train_x), train_w))
            predicted.append(classify(k, distances))
            actual.append(test_w+1)

    cm = confusion_matrix(actual, predicted)
    w = sum([cm[i][j] for i in range(len(cm)) for j in range(len(cm[i])) \
             if i == j]) / len(actual)
    print("K={}, error={}\n".format(k,round(w, 3)), cm, end="\n\n")
    return w
            

def main():
    print()
    fp, train, test = open("iris_data.txt"), [[], [], []], [[], [], []]
    for line in fp:
        line = line.strip().split()
        if line: 
            if len(train[int(line[-1])-1]) < 25:
                train[int(line[-1])-1].append([float(l) for l in line[:-1]])
            else:
                test[int(line[-1])-1].append([float(l) for l in line[:-1]])
    
    ws, ks = [], [i for i in range(1, 25, 4)]
    for k in ks:
        ws.append(knn(k, train, test))
    plot_class(ks, ws)


if __name__ == "__main__":
    main()