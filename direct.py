from cyclone import *
import scipy
from scipy.optimize import direct, Bounds
import matplotlib.pyplot as plt
import time



def objective (x , summand=1):
    if fun_cyclone(x, model="Barth-Muschelknautz")[:2][0] > 1000:
        return fun_cyclone(x, model="Barth-Muschelknautz")[:2][1] * -summand
    else:
        return fun_cyclone(x, model="Barth-Muschelknautz")[:2][1]

def constraint (x):
    
    return fun_cyclone(x, model="Barth-Muschelknautz")[:2][0]

def counter_obj (xx, d):
    d["iterations"] +=1


def direct_perform (fun=objective , initial_guess=np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709 ]), add_penalty=0 , len_tol_func=1e-3):
    start_time = time.time()
    d = {"iterations" : 0}
    cb = lambda xx : counter_obj(xx, d)
    bounds = Bounds([1,2,0.3,0.5,0.5,0.1], [1.5,3,0.5,0.8,0.7,0.3])
    res = direct(func=lambda x: objective(x, add_penalty),  bounds=bounds,len_tol=len_tol_func , callback=cb )
    #print(res.x, res.fun, fun_cyclone(x=res.x)[0])
    end_time = time.time()
    return{
        "E" : res.fun,
        "PL" : constraint(res.x),
        "X" : res.x ,
        "iterations" : d["iterations"],
        "time" : end_time - start_time
    }

    #return(res.fun , fun_cyclone(x=res.x)[0])


def test_differentSummands ():
    i = 0
    while i < 1:
        print(f"I ist {i}")
        direct_perform(add_penalty=i)
        i = i + 0.005


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


    plt.xlabel("Stop Criteria ")
    plt.ylabel("penalty")
    plt.scatter(x1_tols, x2_summands , c=y_eff , cmap='RdBu',vmin=0.95, vmax=0.987)
    plt.xscale('log')
    plt.show()

#create_grid_comparison()
#print(direct_perform())

