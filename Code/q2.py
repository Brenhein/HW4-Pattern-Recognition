import matplotlib.pyplot as plt
import numpy as np
import itertools



IMOX = ["I", "M", "O", "X"]


def plot(imox, E, mu, name):
    """Plots the best 2 features using PCA feature transformation"""
    fig1, ax1 = plt.subplots()
    for w, features in enumerate(imox):
        x1, x2 = [], []
        for f in features:
            y = (E.T).dot(f-mu)
            x1.append(y[0])
            x2.append(y[1])
        ax1.scatter(x1, x2, label=IMOX[w])
    
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_title(name+" Feature Transformation for IMOX")
    ax1.legend()


def main():
    # Gets the data from the IMOX file and put it into a dict
    fp, imox = open("imox_data.txt"), [[], [], [], []]
    for line in fp:
        line = line.split()
        w, features = int(line[-1]), [float(el) for el in line[:-1]]
        imox[w-1].append(features)
    
    # Finds the scatter of the dataset
    imox_all = list(itertools.chain.from_iterable(imox))
    mu_o = np.mean(imox_all, axis=0)
    S = sum([np.outer((el-mu_o), (el-mu_o)) for el in imox_all])
    
    # Uses PCA for feature tranformation
    S_eigval, S_eigvec = np.linalg.eigh(S)
    indices = (-S_eigval).argsort()[:2]
    E = S_eigvec[:, [indices[0], indices[1]]]
    print(E, end="\n\n")
    plot(imox, E, mu_o, "PCA")
    
    # Uses MDA for feature transformation 
    SW, mus = [], []
    for w, features in enumerate(imox):
        mu = np.mean(features, axis=0)
        SW.append(sum([np.outer((el-mu), (el-mu)) for el in features]))
        mus.append(mu)
    SW = sum(SW)     
    print(SW)
    SB = sum([len(imox[i])*(np.outer((mus[i]-mu_o), (mus[i]-mu_o))) \
              for i in range(len(imox))])
    
    # Picks the top eigen vectors besed on their eigen values
    S = np.dot(np.linalg.inv(SW), SB)
    S_eigval, S_eigvec = np.linalg.eigh(S)
    indices = (-S_eigval).argsort()[:2]
    E = S_eigvec[:, [indices[0], indices[1]]]
    print(E, end="\n\n")
    plot(imox, E, mu_o, "MDA")  

if __name__ == "__main__":
    main()
