import numpy as np
n = int(input("How many samples:"))
w1 = np.random.normal(20, 5**.5, n)
w2 = np.random.normal(35, 5**.5, n)
fp1 = open("gen1_" + str(n) + ".txt", "w")
fp2 = open("gen2_" + str(n) + ".txt", "w")
for x in w1:
    fp1.write(str(x)+"\n")
for x in w2:
    fp2.write(str(x)+"\n")
fp1.close()
fp2.close()