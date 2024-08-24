from cyclone import *
import numpy as np
from cmaes import CMA
import cma 
import time


def objective (x, penalty=0.05):
    if constraint(x) > 1000:
        weight = constraint(x) - 1000
        return fun_cyclone(x, model="Barth-Muschelknautz")[:2] + (penalty * 0.01 * weight)
    return fun_cyclone(x, model="Barth-Muschelknautz")[:2]

def constraint (x):
    return fun_cyclone(x, model="Barth-Muschelknautz")[:2]

# Callback
def counter_obj (xx, d ):
    d["iterations"] +=1


def cma_execute (x_start = np.array([1.5,3,0.5,0.8,0.7,0.3])):
    start_time = time.time()
    d = {"iterations" : 0}
    
    cb = lambda xx: counter_obj(xx, d )
    sigma_start = 1.3
    opts = cma.CMAOptions()

    opts.set("bounds", [[1,2,0.3,0.5,0.5,0.1], [1.5,3,0.5,0.8,0.7,0.3]])

    res = cma.fmin( lambda x :  objective(x) , x_start, sigma_start, opts ,callback=cb)
    
    end_time = time.time()

    return {
        "E" : res[1],
        "PL": constraint(res[0]),
        "iterations": res[4],
        "X" : res[0],
        "time" : end_time - start_time 
    }

def help_cma ():
    print(cma.CMAOptions('tol'))

