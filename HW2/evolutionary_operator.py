import copy

import networkx as nx
import numpy as np
import random
from tool_box import Get_P

def Get_fitness(graph,x,n_edges,threshold=0):
    g1=np.where(x==0)[0]
    g2=np.where(x==1)[0]
    fitness = round(nx.cut_size(graph,g1,g2)/n_edges,5)
    return fitness

def One_bit_mutation(x):
    x_new=copy.deepcopy(x)
    idx=random.randint(0,len(x_new)-1)
    x_new[idx]=x_new[idx]^1
    return x_new
def Bit_wise_mutation(x,p,graph,n_edges,ori_fitness):
    x_new=copy.deepcopy(x)
    for num,_ in enumerate(x_new):
        if(random.random()<p):
            x_new[num]=x_new[num]^1
    fitness_new=Get_fitness(graph,x_new,n_edges)
    if(ori_fitness>fitness_new):
        return x
    return x_new

def Heavy_tailed_mutation(x,p,graph,n_edges,ori_fitness):
    x_new=copy.deepcopy(x)
    p=Get_P()
    Len = len(x_new)
    idx = np.random.choice(np.arange(0, Len//2+1, 1), p=p)
    for num, _ in enumerate(x_new):
        if (random.random() * Len < idx):
            x_new[num] = x_new[num] ^ 1
    fitness_new = Get_fitness(graph, x_new, n_edges)
    if (ori_fitness > fitness_new):
        return x
    return x_new

def One_point_crossover(x,y,p):
    if random.random()<p:
        idx=random.randint(0,len(x)-1)
        x_new=np.concatenate((x[:idx],y[idx:]))
        y_new=np.concatenate((y[:idx],x[idx:]))
        return x_new,y_new
    else:
        return x,y

def Two_point_crossover(x,y,p):
    if random.random()<p:
        idx1=random.randint(0,len(x)-1)
        #idx2=random.randint(0,len(x)-1)
        idx2=np.random.choice(np.arange(max(0,idx1-100),min(len(x),idx1+100),1))
        if idx1>idx2:
            idx1,idx2=idx2,idx1
        x_new=np.concatenate((x[:idx1],y[idx1:idx2],x[idx2:]))
        y_new=np.concatenate((y[:idx1],x[idx1:idx2],y[idx2:]))
        return x_new,y_new
    else:
        return x,y

def Cmp_fit(x):
    return x[1]

def Selection_fitness(group, number, game=0.5):
    #g_rk = sorted(group, key=Cmp_fit, reverse=True)
    g_rk=group
    new_group = []
    prob = np.array(list(map(Cmp_fit, g_rk)))
    prob = prob - prob.min()*2/3
    prob = prob / prob.sum()
    for i in range(number):
        index=np.random.choice(np.arange(0,len(group),1),p=prob)
        new_group.append(g_rk[index])
    return new_group

def Survival_with_fitness(group,number):
    g_rk = sorted(group, key=Cmp_fit, reverse=True)
    return g_rk[:number]