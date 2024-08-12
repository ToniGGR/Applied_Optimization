import matplotlib.pyplot as plt
import numpy as np
from neldermead import nedermead_perform
from direct import direct_perform
from randomSearch import random_search
from cma_es import cma_execute
from cobyla import cobyla_perform
from cobyla import cobyqa_perform
from cobyla import slsqp_perform


def compare_iterations (x_test = np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709])):
    np.random.seed(0)
    x_test = np.random.uniform([1,2,0.3,0.5,0.5,0.1] , [1.5,3,0.5,0.8,0.7,0.3])
    algoritms_list = ["random" , "nelder-mead" , "cobyla" , "cobyqa" , "slsqb" , "direct" , "cma_es"]
    iteration_list = []


    #local
    iteration_list.append(random_search(budget=1000)["iterations"])
    iteration_list.append(nedermead_perform(initial_guess=x_test )["iterations"])
    iteration_list.append(cobyla_perform(initial_guess=x_test)["iterations"])
    iteration_list.append(cobyqa_perform(initial_guess=x_test)["iterations"])
    iteration_list.append(slsqp_perform(initial_guess=x_test)["iterations"])
    #global 
    iteration_list.append(direct_perform(initial_guess=x_test)["iterations"])
    iteration_list.append(cma_execute()["iterations"])

    fig, ax = plt.subplots()
    ax.bar(algoritms_list , iteration_list)
    ax.set_xlabel("Algorithms")
    ax.set_ylabel("Amount Iterations")
    ax.set_title(f"Algorithms by Iteration with Start Array {x_test} ")
    plt.show()


def compare_execution_time (x_test = np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709])):
    algoritms_list = ["random" , "nelder-mead" , "cobyla" , "cobyqa" , "slsqb" , "direct" , "cma_es"]
    iteration_list = []

    np.random.seed(0)
    x_test = np.random.uniform([1,2,0.3,0.5,0.5,0.1] , [1.5,3,0.5,0.8,0.7,0.3])

    #local
    iteration_list.append(random_search(budget=1000)["time"])
    iteration_list.append(nedermead_perform(initial_guess=x_test )["time"])
    iteration_list.append(cobyla_perform(initial_guess=x_test)["time"])
    iteration_list.append(cobyqa_perform(initial_guess=x_test)["time"])
    iteration_list.append(slsqp_perform(initial_guess=x_test)["time"])
    #global 
    iteration_list.append(direct_perform(initial_guess=x_test)["time"])
    iteration_list.append(cma_execute()["time"])

    fig, ax = plt.subplots()
    ax.bar(algoritms_list , iteration_list)
    ax.set_xlabel("Algorithms")
    ax.set_ylabel("Time for Calculation")
    ax.set_title(f"Algorithms by Iteration with Start Array {x_test} ")
    plt.show()


def compare_Efficiency (testruns=10):
    algoritms_list = ["random" , "nelder-mead" , "cobyla" , "cobyqa" , "slsqb" , "direct" , "cma_es"]
    list_eff = np.empty((len(algoritms_list),testruns))
    average_eff = []
    failure_ls = [None] * len(algoritms_list)
    cma_es_e = cma_execute()["E"]
    for run in range(testruns):
        np.random.seed(run)
        x_test = np.random.uniform([1,2,0.3,0.5,0.5,0.1] , [1.5,3,0.5,0.8,0.7,0.3])
        list_eff[0][run] = random_search(budget=1000)["E"] * -1
        failure_ls[0] = "g" if random_search(budget=1000)["PL"] <= 1000 and failure_ls[0] != "r"  else "r"
        list_eff[1][run] = nedermead_perform(initial_guess=x_test )["E"] * -1
        failure_ls[1] = "g" if nedermead_perform(initial_guess=x_test )["PL"] <= 1000 and failure_ls[1] != "r" else "r"
        list_eff[2][run] = cobyla_perform(initial_guess=x_test)["E"] * -1
        failure_ls[2] = "g" if cobyla_perform(initial_guess=x_test)["PL"] <= 1001 and failure_ls[2] != "r" else "r"
        list_eff[3][run] = cobyqa_perform(initial_guess=x_test)["E"] * -1
        failure_ls[3] = "g" if cobyqa_perform(initial_guess=x_test)["PL"] <= 1001 and failure_ls[3] != "r" else "r"
        list_eff[4][run] = slsqp_perform(initial_guess=x_test)["E"] * -1
        failure_ls[4] = "g" if slsqp_perform(initial_guess=x_test)["PL"] <= 1000 and failure_ls[4] != "r" else "r"
        list_eff[5][run] = direct_perform(initial_guess=x_test)["E"] * -1
        failure_ls[5] = "g" if direct_perform(initial_guess=x_test)["PL"] <= 1000 and failure_ls[5] != "r" else "r"
        list_eff[6][run] = cma_es_e * -1
        failure_ls[6] = "g"


    get_average =  lambda x : sum(x) / len(x)    

    print(x_test)

    for alg in list_eff:
        average_eff.append(get_average(alg))

    
    fig, ax = plt.subplots()
    ax.bar(algoritms_list , average_eff , color = failure_ls)
    ax.set_xlabel("Algorithms")
    ax.set_ylabel("Max Efficiency")
    ax.set_title(f"Algorithms by Iteration with Start Array {x_test} ")
    plt.show()


def main():
    compare_Efficiency()

if __name__ == "__main__" :
    main()
