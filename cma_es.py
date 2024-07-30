from cyclone import *
import numpy as np
from cmaes import CMA
import cma 

def objective (x):
    penalty = fun_cyclone(x, model="Barth-Muschelknautz")[:2][0] - 1000
    penalty_array = np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709 ])
    if penalty > 0 :
        x = x - (x - penalty_array) * 0.0059 * penalty
    try:
        return fun_cyclone(x, model="Barth-Muschelknautz")[:2][1]
    except:
        return None


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


def quadratic(x1, x2):
    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

def test():
    optimizer = CMA(mean=np.zeros(2), sigma=1.3)

    for generation in range(50):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = quadratic(x[0], x[1])
            solutions.append((x, value))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)


def cma_execute (x_start = np.array([1,2,0.3,0.5,0.5,0.1])):
    sigma_start = 1.3
    opts = cma.CMAOptions()
    opts.set("bounds", [[1,2,0.3,0.5,0.5,0.1], [1.5,3,0.5,0.8,0.7,0.3]])
    cma.fmin(objective, x_start, sigma_start, opts)
    #prints worldwide minimum with 

def help_cma ():
    print(cma.CMAOptions('tol'))


cma_execute()
