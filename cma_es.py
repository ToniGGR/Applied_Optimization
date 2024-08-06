from cyclone import *
import numpy as np
from cmaes import CMA
import cma 
import time


def objective (x, penalty=0.05):
    if constraint(x) > 1000:
        weight = constraint(x) - 1000
        return fun_cyclone(x, model="Barth-Muschelknautz")[:2][1] *  - (penalty * 0.001 * weight)
    return fun_cyclone(x, model="Barth-Muschelknautz")[:2][1]

def constraint (x):
    return fun_cyclone(x, model="Barth-Muschelknautz")[:2][0]

def cma_execute_construction():
    optimizer = CMA(mean=np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709 ]), sigma=0.3)
    opts = cma.CMAOptions()
    opts.set("bounds", [[-2, None], [2, None]])
    #x = initial_guess
    for generation in range(50):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            try :
                value = objective(x)
                solutions.append((x, value))
                print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]}, x3={x[2]}, x4 = {x[3]}, x5={x[4]}, x6 = {x[5]})")
                optimizer.tell(solutions)
                print(solutions)
            except :
                continue


def counter_obj (xx, d):
    d["iterations"] +=1


def cma_execute (x_start = np.array([1.5,3,0.5,0.8,0.7,0.3])):
    start_time = time.time()
    d = {"iterations" : 0}
    cb = lambda xx: counter_obj(xx, d)
    sigma_start = 1.3
    opts = cma.CMAOptions()
    opts.set("bounds", [[1,2,0.3,0.5,0.5,0.1], [1.5,3,0.5,0.8,0.7,0.3]])
    res = cma.fmin(objective, x_start, sigma_start, opts ,callback=cb)
    #print(res[1], res[4] , res[0])
    end_time = time.time()
    return {
        "E" : res[1],
        "PL": constraint(res[0]),
        "iterations": res[4],
        "X" : res[0],
        "time" : end_time - start_time
    }
    #print(d["iterations"])
    #prints worldwide minimum with 

def help_cma ():
    print(cma.CMAOptions('tol'))

#cma_execute()
