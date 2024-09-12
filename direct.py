from cyclone import *
import scipy
from scipy.optimize import direct, Bounds
import matplotlib.pyplot as plt
import time

# Objective with weighted Punishing 
def objective (x , summand=0.02):
    if fun_cyclone(x, model="Barth-Muschelknautz")[0] > 1000:
        return fun_cyclone(x, model="Barth-Muschelknautz")[1] + summand
    else:
        return fun_cyclone(x, model="Barth-Muschelknautz")[1]

def constraint (x):
    return fun_cyclone(x, model="Barth-Muschelknautz")[0]

# Callback Counter Objective
def counter_obj (xx, d , efficiency_log):
    d["iterations"] +=1


def direct_perform (fun=objective , initial_guess=np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709 ]), add_penalty=0.2 , len_tol_func=1e-3):
    start_time = time.time()
    d = {"iterations" : 0}
    efficiency_log = []

    cb = lambda xx : counter_obj(xx, d ,efficiency_log)
    bounds = Bounds([1,2,0.3,0.5,0.5,0.1], [1.5,3,0.5,0.8,0.7,0.3])

    res = direct(objective,  bounds=bounds, len_tol=len_tol_func, callback=cb)

    end_time = time.time()

    return{
        "E" : res.fun,
        "PL" : constraint(res.x),
        "X" : res.x,
        "iterations" : d["iterations"],
        "time" : end_time - start_time,
        "progress_list" : efficiency_log
    }

# For Testing Purpose
def test_differentSummands ():
    i = 0
    while i < 1:
        print(f"I ist {i}")
        direct_perform(add_penalty=i)
        i = i + 0.005

# Grip Comparison for Hyperparameter Tuning
def create_grid_comparison():
    tols = [1e-1,1e-2,1e-3,1e-4, 1e-5, 1e-6]
    x1_tols = []
    x2_summands = []
    y_eff = []
    Y_pl = []

    tol = 0.001

    for tol in tols :
        summand = 0
        while summand < 0.1:
            result = direct_perform(add_penalty=summand, len_tol_func=tol )
            x1_tols.append(tol)
            x2_summands.append(summand)
            y_eff.append(-1 if result["PL"] > 1000 else (result["E"] * -1))
            summand = summand + 0.005


    plt.xlabel("Stop Criteria")
    plt.ylabel("penalty")
    plt.scatter(x1_tols, x2_summands, c=y_eff, cmap='RdBu', vmin=0.95, vmax=0.987)
    plt.xscale('log')
    plt.show()
 



