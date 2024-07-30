from cyclone import *
import scipy
from scipy.optimize import minimize


def objective (x):
    penalty = fun_cyclone(x, model="Barth-Muschelknautz")[:2][0] - 1000
    penalty_array = np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709 ])
    #penalty_array = np.array([1,2.5,0.3,0.5,0.6,0.14])
    if penalty > 0 :
        x = x - ((x - penalty_array) * 0.00291 * penalty)
    return fun_cyclone(x, model="Barth-Muschelknautz")[:2][1]


def nedermead_perform (fun=objective , initial_guess=np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709 ])):
    result = minimize(fun , initial_guess , method="Nelder-Mead" , bounds=[(1,1.5) , (2,3) , (0.3,0.5) , (0.5,0.8), (0.5,0.7),(0.1,0.3) ], options={'xatol': 1e-1, 'disp': True , 'solver':'minimize_scalar'} )
     
    computedsolutionOptimizationX = result.x

    nelder_mead_result = fun_cyclone(computedsolutionOptimizationX)
    print("Optimization Result E:" , nelder_mead_result[1])
    print("Optimization Result PL:" , nelder_mead_result[0])
    print("Optimization Result X:" ,computedsolutionOptimizationX)

    return {
        "E" : nelder_mead_result[1],
        "PL" : nelder_mead_result[0],
        "X" : computedsolutionOptimizationX
    }




nedermead_perform(initial_guess=np.array([1,2.5,0.3,0.5,0.6,0.14]))

