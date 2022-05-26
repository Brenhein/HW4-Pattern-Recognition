import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp


def hn(gen, h, x):
    """For a given set of points (from a normal distribution), the resulting
    estimatiated denisity for a given x using a gaussian kernel function"""
    px, dist = 0, sp.norm(0,1)
    for v in gen: # estimate the density of x using generated points
        px += dist.pdf((x-v)/h)
    return px / (len(gen)*h)
    

def plot_density(gen, h, i):
    """For 56 points(0-55), each points density is estimated using a guassian 
    kernel funciton, and the reuslting 56 points are plotted"""
    x_s = np.linspace(0, 56, 56)
    parzen = [hn(gen, h, x) for x in x_s]  # estimate the density of these points
    
    fig, ax = plt.subplots()   
    ax.set_xlabel("x1")
    ax.set_ylabel("P(x)")
    ax.set_title("Gaussian Parzen Window (h={}, n={})".format(h, int(i)*2))    
    ax.plot(x_s, parzen)


def main():
    for i in ["100", "500", "1000"]:
        gen = [float(l) for l in open("gen1_{}.txt".format(i))] + \
              [float(l) for l in open("gen2_{}.txt".format(i))]
        plot_density(gen, .01, i)
        plot_density(gen, .1, i)
        plot_density(gen, 1, i)
        plot_density(gen, 10, i)
    

if __name__=="__main__":
    main()