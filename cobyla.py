from cyclone import *
import scipy
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint


def objective (x):

    return fun_cyclone(x, model="Barth-Muschelknautz")[:2][1]

def constraint (x):
    
    return fun_cyclone(x, model="Barth-Muschelknautz")[:2][0]


def cobyla_perform (fun=objective , initial_guess=np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709 ])):
    nonlinear_constraint = NonlinearConstraint(fun=constraint, lb=0 , ub=1000 )

    result = minimize(fun , initial_guess , method="COBYLA" , bounds=[(1,1.5) , (2,3) , (0.3,0.5) , (0.5,0.8), (0.5,0.7),(0.1,0.3) ], options={'xatol': 1e-1, 'disp': True} , constraints=nonlinear_constraint )
     
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

def cobyqa_perform (fun=objective , initial_guess=np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709 ])):
    nonlinear_constraint = NonlinearConstraint(fun=constraint, lb=0 , ub=1000 )

    result = minimize(fun , initial_guess , method="COBYQA" , bounds=[(1,1.5) , (2,3) , (0.3,0.5) , (0.5,0.8), (0.5,0.7),(0.1,0.3) ], options={'xatol': 1e-1, 'disp': True} , constraints=nonlinear_constraint )
     
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

def slsqp_perform (fun=objective , initial_guess=np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709 ])):
    nonlinear_constraint = NonlinearConstraint(fun=constraint, lb=0 , ub=1000 )

    result = minimize(fun , initial_guess , method="SLSQP" , bounds=[(1,1.5) , (2,3) , (0.3,0.5) , (0.5,0.8), (0.5,0.7),(0.1,0.3) ], options={'xatol': 1e-1, 'disp': True} , constraints=nonlinear_constraint )
     
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

cobyla_perform(initial_guess=np.array([1,2.5,0.3,0.5,0.6,0.14]))
cobyqa_perform(initial_guess=np.array([1,2.5,0.3,0.5,0.6,0.14]))
slsqp_perform(initial_guess=np.array([1.5,3,0.5,0.8,0.7,0.3]))
