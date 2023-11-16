import networkx as nx
import numpy as np
import copy
import time
import argparse
import matplotlib
import matplotlib.pyplot as plt

from graph_generator import graph_generator
from evolutionary_operator import Bit_wise_mutation, One_bit_mutation, Get_fitness, One_point_crossover, \
    Two_point_crossover, Selection_fitness, Survival_with_fitness, Heavy_tailed_mutation
from tool_box import Generate_random_binary_string, Init_P, Get_P


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-type', type=str, help='graph type', default='gset')
    parser.add_argument('--n-nodes', type=int, help='the number of nodes', default=1000)
    parser.add_argument('--n-d', type=int, help='the number of degrees for each node', default=10)
    parser.add_argument('--T', type=int, help='the number of fitness evaluations', default=2000)
    parser.add_argument('--seed-g', type=int, help='the seed of generating regular graph', default=1)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--gset-id', type=int, default=1)
    parser.add_argument('--sigma', type=float, help='hyper-parameter of mutation operator', default=.1)
    parser.add_argument('--size', type=int, help='the size of the group', default=1)
    # parser.add_argument('--lamda', type=int, help='the number per parent selection', default=2)
    parser.add_argument('--prob_m', type=int, help='the propobility of bit-wise mutation', default=0.004)
    parser.add_argument('--prob_c', type=int, help='the propobility of one-point crossover', default=0.6)
    # parser.add_argument('--gama', type=int, help='for FPS', default=0.5)
    args = parser.parse_known_args()[0]
    return args


def real_number_format(args=get_args()):
    graph, n_nodes, n_edges = graph_generator(args)
    np.random.seed(args.seed)
    x = np.random.rand(n_nodes)
    best_fitness = Get_fitness(graph, x, n_edges)  # After Normalization
    best_T = []
    best_fit_list = []
    for i in range(args.T):  # iteration for T times
        tmp = x + np.random.randn(
            n_nodes) * args.sigma  # In each round, use a mutation operator with a σ times standard normal distribution
        tmp_fitness = Get_fitness(graph, tmp, n_edges)
        if tmp_fitness > best_fitness:
            x, best_fitness = tmp, tmp_fitness
            print(i, best_fitness)
        best_T.append(i)
        best_fit_list.append(best_fitness)
    # plot_performance(best_T, best_fit_list, label="baseline:real-number")


def binary_string_individual(args=get_args()):
    graph, n_nodes, n_edges = graph_generator(args)
    x = Generate_random_binary_string(n_nodes)
    best_fitness = Get_fitness(graph, x, n_edges)

    for i in range(args.T):
        tmp = Bit_wise_mutation(x, args.prob_m)
        tmp_fitness = Get_fitness(graph, tmp, n_edges)
        if tmp_fitness > best_fitness:
            x, best_fitness = tmp, tmp_fitness
            print(i, best_fitness)


def SEA(args=get_args()):
    graph, n_nodes, n_edge = graph_generator(args)
    group = []
    max_fitness = 0
    Init_P(n_nodes)
    for i in range(args.size):
        x = Generate_random_binary_string(n_nodes)
        fitness = Get_fitness(graph, x, n_edge)
        group.append((x, fitness))

    iterations = []
    best_results = []
    for _ in range(args.T):
        old_group = Selection_fitness(group, args.size)
        new_group = copy.deepcopy(group)

        #for i in range(args.size):
       #     idx, idy = np.random.choice(args.size, 2)
       #     x, y = old_group[idx][0], old_group[idy][0]
       #     x, y = Two_point_crossover(x, y, args.prob_c)
        #    x = Bit_wise_mutation(x, args.prob_m, graph, n_edge,old_group[idx][1])
        #    y = Bit_wise_mutation(y, args.prob_m, graph, n_edge,old_group[idy][1])
       #     fitness_x = Get_fitness(graph, x, n_edge)
       #     fitness_y = Get_fitness(graph, y, n_edge)
       #     max_fitness = max(max_fitness, fitness_x, fitness_y)
       #     new_group.append((x, fitness_x))
        #    new_group.append((y, fitness_y))
        Group = []

        for i in new_group:
            x= Heavy_tailed_mutation(i[0],args.prob_m,graph,n_edge,i[1])
            new_fitness=Get_fitness(graph,x,n_edge)
            max_fitness=max(max_fitness,new_fitness)
            Group.append((x,new_fitness))
        group = Survival_with_fitness(Group, args.size)
        iterations.append(_)
        best_results.append(max_fitness)
        if(_ % 100 == 0):
            print(_, max_fitness)

    # 绘制曲线
    return iterations, best_results

def Run_with_different_graph(args=get_args()):
    param_sets = []
    for i in range(1, 11):
        param_sets.append("G"+str(i))
    results = []
    for i in range(1, 11):
        args.gset_id = i
        iterations, best_results = SEA(args)
        results.append({'params': param_sets[i-1], 'iterations': iterations, 'best_results': best_results})

    # 绘制曲线
    for result in results:
        plt.plot(result['iterations'], result['best_results'], label=str(result['params']))

    # 设置图例位置为右上角
    plt.legend(title='Parameter Sets', loc='lower right')

    plt.title('Evolution of Best Results Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Best Results')
    plt.grid(True)
    plt.show()
def main(args=get_args()):
    # real_number_format(args)
    #binary_string_individual(args)
    #SEA(args)
    Run_with_different_graph(args)


if __name__ == '__main__':
    main()
