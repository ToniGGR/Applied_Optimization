import matplotlib.pyplot as plt
import numpy as np
from neldermead import nedermead_perform
from direct import direct_perform
from randomSearch import random_search
from cma_es import cma_execute
from cobyla import cobyla_perform
from cobyla import cobyqa_perform
from cobyla import slsqp_perform
from cyclone import fun_cyclone


def compare_iterations (x_test = np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709]) ,  testruns = 10):
    algoritms_list = ["random" , "nelder-mead" , "cobyla" , "cobyqa" , "slsqb" , "direct" , "cma_es"]
    iteration_list = range(len(algoritms_list))
    dict_iter = {"random" : [] , "nelder-mead": [] , "cobyla": [] , "cobyqa" : [], "slsqb" : [] , "direct":[] , "cma_es": []  }

    for _ in range(testruns):
        np.random.seed(_ * 7)
        x_test = np.random.uniform([1,2,0.3,0.5,0.5,0.1] , [1.5,3,0.5,0.8,0.7,0.3])
        #local
        dict_iter["random"].append(random_search(budget=1000)["iterations"])
        dict_iter["nelder-mead"].append(nedermead_perform(initial_guess=x_test )["iterations"])
        dict_iter["cobyla"].append(cobyla_perform(initial_guess=x_test)["iterations"])
        dict_iter["cobyqa"].append(cobyqa_perform(initial_guess=x_test)["iterations"])
        dict_iter["slsqb"].append(slsqp_perform(initial_guess=x_test)["iterations"])
        #global 
        dict_iter["direct"].append(direct_perform(initial_guess=x_test)["iterations"])
        dict_iter["cma_es"].append(cma_execute()["iterations"])
        
    fig, ax = plt.subplots()
    plt.boxplot(dict_iter.values() , tick_labels=algoritms_list)
    plt.title(f"Amount Iterations \n {testruns} random runs")
    plt.xlabel(f"\n Algorithmn")
    plt.ylabel("Amount of Objective Function Calls")
    plt.show()


def compare_execution_time (x_test = np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709]) , testruns = 10):
    algoritms_list = ["random" , "nelder-mead" , "cobyla" , "cobyqa" , "slsqb" , "direct" , "cma_es"]
    dict_time = {"random" : [] , "nelder-mead": [] , "cobyla": [] , "cobyqa" : [], "slsqb" : [] , "direct":[] , "cma_es": []  }

    # Fill Dictionary
    for _ in range(testruns):
        np.random.seed(_ * 52)
        x_test = np.random.uniform([1,2,0.3,0.5,0.5,0.1] , [1.5,3,0.5,0.8,0.7,0.3])
        #local
        dict_time["random"].append(random_search(budget=100000)["time"])
        dict_time["nelder-mead"].append(nedermead_perform(initial_guess=x_test )["time"])
        dict_time["cobyla"].append(cobyla_perform(initial_guess=x_test)["time"])
        dict_time["cobyqa"].append(cobyqa_perform(initial_guess=x_test)["time"])
        dict_time["slsqb"].append(slsqp_perform(initial_guess=x_test)["time"])
        #global 
        dict_time["direct"].append(direct_perform(initial_guess=x_test)["time"])
        dict_time["cma_es"].append(cma_execute()["time"])

    # Plot Boxplots
    fig, ax = plt.subplots()
    plt.boxplot(dict_time.values() , tick_labels=algoritms_list)
    plt.title(f"Execution Time \n {testruns} random runs")
    plt.xlabel(f"\nAlgorithms")
    plt.ylabel("execution time in ms")
    plt.show()

#Check if input is x valid (Doesnt Violate Constraints)
def check_valid (x):
    lower=[1,2,0.3,0.5,0.5,0.1]
    upper=[1.5,3,0.5,0.8,0.7,0.3]
    #Check if x is in Bounds
    check_bounds =  all( lower[i] <= x[i] and upper[i] >= x[i] for i in range(len(x)))

    # Get Pressureloss
    check_pl = fun_cyclone(x)[0]
    
    return check_bounds and (check_pl <= 1000)

# Get Constraint Violations 
def compare_Validiy (testruns=10):
    algoritms_list = ["random", "nelder-mead" , "cobyla" , "cobyqa" , "slsqb" ]
    amount_fail = [ 0 for x in algoritms_list]

    for run in range(testruns):
        np.random.seed(run * 12)
        x_test = np.random.uniform([1,2,0.3,0.5,0.5,0.1] , [1.5,3,0.5,0.8,0.7,0.3])
        amount_fail[0] = amount_fail[0] +  1 if not check_valid(list(random_search(budget=80000)["X"][0])) else amount_fail[0] 
        
        amount_fail[1] = amount_fail[1] + 1 if  not check_valid(nedermead_perform(initial_guess=x_test )["X"]) else amount_fail[1]

        amount_fail[2] = amount_fail[2] + 1 if not check_valid(cobyla_perform(initial_guess=x_test )["X"]) else amount_fail[2]

        amount_fail[3] = amount_fail[3] + 1 if not check_valid(cobyqa_perform(initial_guess=x_test )["X"]) else  amount_fail[3]

        amount_fail[4] = amount_fail[4] + 1 if not check_valid(slsqp_perform(initial_guess=x_test )["X"]) else  amount_fail[4]


    fig, ax = plt.subplots()
    ax.bar(algoritms_list, amount_fail , color="red")
    ax.bar(algoritms_list, testruns - np.array(amount_fail)  , bottom=amount_fail , color="green")
    ax.set_ylabel("Amount of constraint violation")
    ax.set_title(f"{testruns} testruns")
    plt.show()

# Get Average Efficiency of each Algorithm
def compare_Efficiency (testruns=10):
    algoritms_list = ["random" , "nelder-mead" , "cobyla" , "cobyqa" , "slsqb" , "direct" , "cma_es"]
    list_eff = np.empty((len(algoritms_list),testruns))
    average_eff = []
    failure_ls = [None] * len(algoritms_list)

    #Execute CMA as it is deterministic
    cma_es_e = cma_execute()["E"]
    for run in range(testruns):
        np.random.seed(run * 5)
        x_test = np.random.uniform([1,2,0.3,0.5,0.5,0.1] , [1.5,3,0.5,0.8,0.7,0.3])
        list_eff[0][run] = random_search(budget=80000)["E"] * -1
        failure_ls[0] = "g" if True  else "r"
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

    # Get Average Lambda from a dynamic list
    get_average =  lambda x : sum(x) / len(x)    

    # Get Average of each Algorithm
    for alg in list_eff:
        average_eff.append(get_average(alg))
    
    fig, ax = plt.subplots()
    ax.bar(algoritms_list , average_eff , color = failure_ls)
    ax.set_xlabel("Algorithms")
    ax.set_ylabel("Max Efficiency")
    ax.set_title(f"Algorithms by Iteration with Start Array {average_eff} ")
    plt.show()

def compare_scatter (testruns=10):
    algoritms_list = ["random" , "nelder-mead" , "cobyla" , "cobyqa" , "slsqb" , "direct" , "cma_es"]
    list_eff = np.empty((len(algoritms_list),testruns))
    average_eff = []

    # Perform CMA as it is deterministic
    cma_es_e = cma_execute()["E"]

    for run in range(testruns):
        np.random.seed(run * 4)
        x_test = np.random.uniform([1,2,0.3,0.5,0.5,0.1] , [1.5,3,0.5,0.8,0.7,0.3])
        list_eff[0][run] = random_search(budget=80000)["E"] * -1
        
        list_eff[1][run] = nedermead_perform(initial_guess=x_test )["E"] * -1
        
        list_eff[2][run] = cobyla_perform(initial_guess=x_test)["E"] * -1
        
        list_eff[3][run] = cobyqa_perform(initial_guess=x_test)["E"] * -1
        
        list_eff[4][run] = slsqp_perform(initial_guess=x_test)["E"] * -1
        
        list_eff[5][run] = direct_perform(initial_guess=x_test)["E"] * -1
        
        list_eff[6][run] = cma_es_e * -1

    # Lambda to get Average of a dynamic list
    get_average =  lambda x : sum(x) / len(x)    

    for alg in list_eff:
        average_eff.append(get_average(alg))
    
    fig, ax = plt.subplots( 1, (len(algoritms_list)))

    for index in range(len(algoritms_list)) :
        ax[index].boxplot(list_eff[index] , showmeans=True)
        ax[index].set_xlabel("Algorithms")
        ax[index].set_ylabel("Efficiency")
        ax[index].set_title(f"Algorithm {algoritms_list[index]}")
        ax[index].set_ylim([0.95 , 1])
    plt.show()




def compare_progress () :
    # Create Random Seed
    np.random.seed(90)
    x_test = np.random.uniform([1,2,0.3,0.5,0.5,0.1] , [1.5,3,0.5,0.8,0.7,0.3])

    # Get Dictionary of each Algorithm
    nelder_mead_perf_list = nedermead_perform (initial_guess=x_test)
    slsqp_perf_list = slsqp_perform(initial_guess=x_test)
    cobyqa_perf_list = cobyqa_perform(initial_guess=x_test)
    cobyla_perf_list = cobyla_perform(initial_guess=x_test)
    random_perf_list = random_search(budget=10000)


    fig, ax = plt.subplots( 2,1 )

    # Efficiency Progress of each Algorithm
    ax[0].set_xlabel("Run")
    ax[0].set_ylabel("Efficiency")
    ax[0].plot(range(len(nelder_mead_perf_list["progress_list"])),nelder_mead_perf_list["progress_list"] ,  '-o' , markersize=3.2 , linewidth = 1.1    )
    ax[0].plot(range(len(slsqp_perf_list["progress_list"])),slsqp_perf_list["progress_list"] , '-o' , c="g" , markersize=3.2 , linewidth = 1.1  )
    ax[0].plot(range(len(cobyqa_perf_list["progress_list"])),cobyqa_perf_list["progress_list"] , '-o' , c="r" ,markersize=3.2 , linewidth = 1.1 )
    ax[0].plot(range(len(cobyla_perf_list["progress_list"])),cobyla_perf_list["progress_list"]  , '-o' , c="black" , markersize=3.2 , linewidth = 1.1)
    ax[0].plot(range(len(random_perf_list["progress_list"])),random_perf_list["progress_list"]  , '-o', c="grey" , markersize=3.2 , linewidth = 1.1)
    ax[0].legend(["Nelder Mead" , "SLSQP" , "COBYQA" , "COBYLA" , "RANDOM"])
    

    # Pressure Loss Progress of each Algorithm
    ax[1].set_xlabel("Run")
    ax[1].set_ylabel("Pressure Delta")
    ax[1].set_ylim (0, 1000)
    ax[1].plot(range(len(nelder_mead_perf_list["pressure_list"])),nelder_mead_perf_list["pressure_list"] ,  '-o'  , markersize=3.2 , linewidth = 1.1  )
    ax[1].plot(range(len(slsqp_perf_list["pressure_list"])),slsqp_perf_list["pressure_list"] , '-o' , c="g"  ,markersize=3.2 , linewidth = 1.1 )
    ax[1].plot(range(len(cobyqa_perf_list["pressure_list"])),cobyqa_perf_list["pressure_list"] , '-o' , c="r" , markersize=3.2 , linewidth = 1.1)
    ax[1].plot(range(len(cobyla_perf_list["pressure_list"])),cobyla_perf_list["pressure_list"]  , '-o' , c="black" , markersize=3.2 , linewidth = 1.1)
    ax[1].plot(range(len(random_perf_list["pressure_list"])),random_perf_list["pressure_list"]  , '-o', c="grey" , markersize=3.2 , linewidth = 1.1)
    ax[1].legend(["Nelder Mead" , "SLSQP" , "COBYQA" , "COBYLA" , "RANDOM"])
    plt.show()

def main():
    ...

if __name__ == "__main__" :
    main()
