from cyclone import *
import scipy
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import matplotlib.pyplot as plt
import time
import optuna

def objective (x , penalty=0.85):
    ### NORMAL PENALTY
    #if constraint(x) > 1000:
    #    return fun_cyclone(x, model="Barth-Muschelknautz")[:2][1] *  - (penalty)
    #return fun_cyclone(x, model="Barth-Muschelknautz")[:2][1]

    ### WEIGHTED PENALTY
    if constraint(x) > 1000:
        weight = constraint(x) - 1000
        return fun_cyclone(x, model="Barth-Muschelknautz")[1] + (penalty * 0.001 * weight )
    return fun_cyclone(x, model="Barth-Muschelknautz")[1]

# Pressure Loss Function
def constraint (x):
    return fun_cyclone(x, model="Barth-Muschelknautz")[0]

#Counter of Objective Function
def counter_obj (xx, d , efficiency_log , pressure_log):
    d["iterations"] +=1
    efficiency_log.append(fun_cyclone(xx)[1])
    pressure_log.append(fun_cyclone(xx)[0])
    
# Nelder Mead Execution
def nedermead_perform (fun=objective , initial_guess=np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709 ]) , penalty=0):
    start_time = time.time()
    d = {"iterations" : 0}
    efficiency_log = []
    pressure_log = []

    #Callbakc to count iterations
    cb = lambda xx : counter_obj(xx,d , efficiency_log , pressure_log)
    
    #Nonlinear Constraint Pressure loss
    nonlinear_constraint = NonlinearConstraint(fun=constraint, lb=0 , ub=1000 )
    
    #Perfrom Nelder Mead
    result = minimize(fun  , initial_guess , method="Nelder-Mead" , bounds=[(1,1.5) , (2,3) , (0.3,0.5) , (0.5,0.8), (0.5,0.7),(0.1,0.3) ], options={'xatol': 1e-1, 'disp': True , 'solver':'minimize_scalar'}, constraints=nonlinear_constraint  , callback=cb  )

    # X Result    
    computedsolutionOptimizationX = result.x

    # Get Efficiency and Pressyre Loss
    nelder_mead_result = fun_cyclone(computedsolutionOptimizationX)
    
    #Stop Timer
    end_time = time.time()

    return {
        "E" : nelder_mead_result[1],
        "PL" : nelder_mead_result[0],
        "X" : computedsolutionOptimizationX,
        "iterations" : d["iterations"] ,
        "time" : end_time - start_time ,
        "progress_list" : efficiency_log ,
        "pressure_list" : pressure_log
    }

# Test 50 Runs of Nelder Mead
def examine_x():
    best_x = [0,0,0,0,0,0]
    runs = []
    y_eff = []
    best_P = 0
    counter = 0
    max_predict = 0
    
    for _ in range(50):
        i = np.random.uniform(np.array([1,2,0.3,0.5,0.5,0.1] ),np.array([1.5,3,0.5,0.8,0.7,0.3]))
        try:
            res = nedermead_perform(initial_guess= i)
            counter = counter +1
            runs.append(counter)
            y_eff.append([res["E"] * - 1 if (res["PL"] < 1000) else -1])
            if max_predict > res["E"] and res["PL"] < 1000:
                max_predict = res["E"]
                best_x = res["X"]
                best_P = res["PL"]
        except:
            continue
        
    print(max_predict , best_x , best_P)

    plt.xlabel("Stop Criteria ")
    plt.ylabel("penalty")
    plt.scatter(runs, y_eff  )
    plt.show()


# Tuning for Nelder Mead
def tuning ():
    # Create Study
    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_objective, n_trials=100)

    # Beste gefundene Parameter
    best_trial = study.best_trial
    print(f"Beste Parameter: {best_trial.params}")
    print(f"Beste Zielfunktion: {best_trial.value}")
    

# Optuna Function
def optuna_objective (trial) :
    # Hyperparameter, to be optimized
    max_iter = trial.suggest_int('maxiter', 50, 900)
    xatol = trial.suggest_loguniform('xatol' , 1e-8 , 1e-3)
    fatol = trial.suggest_loguniform('fatol' , 1e-8 , 1e-3)
    penalty = trial.suggest_loguniform ("penalty" , 1e-2 , 1)

    # Start of optimization
    initial_guess = np.array([1.5,3,0.5,0.8,0.7,0.3])

    nonlinear_constraint = NonlinearConstraint(fun=constraint, lb=0 , ub=1000 )

    # Scipy minimize-Call
    result = minimize(objective(penalty= penalty) , initial_guess , method="Nelder-Mead" , bounds=[(1,1.5) , (2,3) , (0.3,0.5) , (0.5,0.8), (0.5,0.7),(0.1,0.3) ], options={'xatol': xatol, 'fatol' : fatol ,  'disp': True , 'solver':'minimize_scalar', 'maxiter': max_iter}, constraints=nonlinear_constraint)
    return result.fun

if __name__ == "__main__":
    ...

