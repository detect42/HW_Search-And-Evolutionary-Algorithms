import numpy as np
import random
import time

def Generate_random_binary_string(length):
    return np.array([int(random.choice([0, 1])) for _ in range(length)])

P=[]
def Init_P(n):
    P.clear()
    B=1.5
    sum=0
    for i in range(n//2):
        sum =sum + (i+1)**(-B)
    P.append(0)
    for i in range(n//2):
        P.append(((i+1)**(-B))/sum)
    print(P)

def Get_P():
    return P
