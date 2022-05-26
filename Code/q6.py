from operator import itemgetter
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy.stats as sp


def hn(train, h, x):
    """For a given set of points (from a normal distribution), the resulting
    estimatiated denisity for a given x using a gaussian kernel function"""
    dist, densities = sp.multivariate_normal([0,0], [[1,0], [0,1]]), []
    for w, patterns in enumerate(train): # estimate the density of x using generated points
        px = 0
        for v in patterns:
            px += dist.pdf((np.subtract(x,v))/h)
        densities.append((px / (len(train)*h*h), w+1))
    return densities


def error(predicted, actual):
    """Gets the confusion matrix and the error rate for the test set"""
    cm = confusion_matrix(actual, predicted)
    w = sum([cm[i][j] for i in range(len(cm)) for j in range(len(cm[i])) if i != j])
    print(cm, "\n")
    print("Error:", w/len(actual), "\n\n")


def compare_results(test, gaus1, gaus2, gaus3):
    """finds the predicted class given the different gaussian distribution"""
    predicted, actual = [], []
    for w, patterns in enumerate(test):
        for pat in patterns:
            best = max([(gaus1.pdf(pat), 1), (gaus2.pdf(pat), 2), 
                        (gaus3.pdf(pat), 3)], key=itemgetter(0))
            predicted.append(best[1])
            actual.append(w + 1) 
    error(predicted, actual)


def get_confuse(test, mus, covs):
    """Does the set up of the data to get the actual and predicted values of
    the classifier using the test set"""
    mu1, mu2, mu3 = mus[0], mus[1], mus[2]
    cov1, cov2, cov3 = covs[0], covs[1], covs[2]
    gaus1 = sp.multivariate_normal(mu1, cov1)
    gaus2 = sp.multivariate_normal(mu2, cov2)
    gaus3 = sp.multivariate_normal(mu3, cov3)
    compare_results(test, gaus1, gaus2, gaus3)


def part_b(train, test):
    mus, covs = [], []
    for w, patterns in enumerate(train):
        mus.append(np.mean(patterns, axis=0))
        covs.append(np.cov(patterns, rowvar=False))
    get_confuse(test, mus, covs)

def part_c(train, test):
    predicted, actual = [], []
    for w, patterns in enumerate(test):
        for pat in patterns:
            d_est = hn(train, 1, pat)
            best = max(d_est, key=itemgetter(0))[1]
            predicted.append(best)
            actual.append(w+1)
    error(predicted, actual)


def main():
    fp = open("hw04_data.txt")
    train, test = [[], [], []], [[], [], []]
    
    for pat_s in fp:
        pat = [float(f) for f in pat_s.split()[:-1]]
        w = int(pat_s[-2])-1
        if len(train[w]) > 249:
            test[w].append(pat)
        else:
            train[w].append(pat)
            
    get_confuse(test, [[0,0], [10,0], [5,5]],
                [[[4,0],[0,4]], [[4,0],[0,4]], [[5,0],[0,5]]])
    part_b(train, test)
    part_c(train, test)

if __name__ == "__main__":
    main()