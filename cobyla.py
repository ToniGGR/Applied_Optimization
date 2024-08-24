import optuna
from cyclone import *
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import time
import matplotlib.pyplot as plt

# No Constraint Check due to Constraint in Scipy Minimize
def objective (x):
    return fun_cyclone(x, model="Barth-Muschelknautz")[1]

def constraint (x):
    return fun_cyclone(x, model="Barth-Muschelknautz")[0]

# Callback Counter
def counter_obj (xx, d , efficiency_log , pressure_log):
    d["iteration"] +=1
    efficiency_log.append(fun_cyclone(xx)[1])
    pressure_log.append(fun_cyclone(xx)[0])

def cobyla_perform (fun=objective , initial_guess=np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709 ])):
    start_time = time.time()
    nonlinear_constraint = NonlinearConstraint(fun=constraint, lb=0 , ub=999  )
    d = {"iteration" : 0}
    efficiency_log = []
    pressure_log = []

    # Callback 
    cb = lambda xx : counter_obj(xx, d , efficiency_log , pressure_log)

    # Perform COBYLA
    result = minimize(fun , initial_guess , method="COBYLA" , bounds=[(1,1.5), (2,3), (0.3,0.5), (0.5,0.8), (0.5,0.7),(0.1,0.3) ], options={'disp': True , "catol" : 9.928618655741073e-05} , constraints=nonlinear_constraint , callback=cb)
     
    computedsolutionOptimizationX = result.x

    nelder_mead_result = fun_cyclone(computedsolutionOptimizationX)
    end_time = time.time()
    
    return {
        "E" : nelder_mead_result[1],
        "PL" : nelder_mead_result[0],
        "X" : computedsolutionOptimizationX,
        "iterations" : d["iteration"],
        "time" : end_time - start_time ,
        "progress_list" : efficiency_log ,
        "pressure_list" : pressure_log
    }

def cobyqa_perform (fun=objective , initial_guess=np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709 ])):
    start_time = time.time()
    nonlinear_constraint = NonlinearConstraint(fun=constraint, lb=0 , ub=999 )
    d = {"iteration" : 0}
    efficiency_log = []
    pressure_log = []

    cb = lambda xx : counter_obj(xx, d , efficiency_log , pressure_log)

    result = minimize(objective , initial_guess , method="COBYQA" , bounds=[(1,1.5) , (2,3) , (0.3,0.5) , (0.5,0.8), (0.5,0.7) , (0.1,0.3) ], options={'maxiter': 718, 'max_fev' : 66 , 'initial_tr_radius': 0.00025993027876526067 ,'disp': True , 'solver':'minimize_scalar'}, constraints=nonlinear_constraint , callback=cb )
     
    computedsolutionOptimizationX = result.x

    nelder_mead_result = fun_cyclone(computedsolutionOptimizationX)

    end_time = time.time()
    return {
        "E" : nelder_mead_result[1],
        "PL" : nelder_mead_result[0],
        "X" : computedsolutionOptimizationX,
        "iterations" : d["iteration"],
        "time" : end_time - start_time,
        "progress_list" : efficiency_log ,
        "pressure_list" : pressure_log
    }

def slsqp_perform (fun=objective , initial_guess=np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709 ])):
    start_time = time.time()
    nonlinear_constraint = NonlinearConstraint(fun=constraint, lb=0 , ub=1000 )
    d = {"iteration" : 0}
    efficiency_log = []
    pressure_log = []

    cb = lambda xx : counter_obj(xx, d , efficiency_log ,pressure_log)
    
    result = minimize(objective , initial_guess , method="SLSQP" , bounds=[(1,1.5) , (2,3) , (0.3,0.5) , (0.5,0.8), (0.5,0.7),(0.1,0.3) ], options={'maxiter': 314, 'eps': 9.995619041053416e-08 } , constraints=nonlinear_constraint, callback=cb )
     
    computedsolutionOptimizationX = result.x

    nelder_mead_result = fun_cyclone(computedsolutionOptimizationX)

    end_time = time.time()
    return {
        "E" : nelder_mead_result[1],
        "PL" : nelder_mead_result[0],
        "X" : computedsolutionOptimizationX,
        "iterations" : d["iteration"],
        "time" : end_time - start_time,
        "progress_list" : efficiency_log ,
        "pressure_list" : pressure_log
    }

# Check 50 Results from the Algorithms
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
            res = cobyla_perform(initial_guess= i )
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

# Template Function for tuning 
def tuning ():
    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_objective, n_trials=100)

    # Beste gefundene Parameter und Ergebnis
    best_trial = study.best_trial
    print(f"Beste Parameter: {best_trial.params}")
    print(f"Beste Zielfunktion: {best_trial.value}")
    

def optuna_objective (trial) :
    # Hyperparameter, die optimiert werden sollen
    max_iter = trial.suggest_int('maxiter', 50, 900)
    
    eps = trial.suggest_loguniform('eps' , 1e-8 , 1e-7)
    


    # Startpunkt der Optimierung
    initial_guess = np.array([1.5,3,0.5,0.8,0.7,0.3])

    nonlinear_constraint = NonlinearConstraint(fun=constraint, lb=0 , ub=1000 )

    # Scipy minimize-Aufruf
    result = minimize(objective , initial_guess , method="SLSQP" , bounds=[(1,1.5) , (2,3) , (0.3,0.5) , (0.5,0.8), (0.5,0.7),(0.1,0.3) ], options={'maxiter': max_iter, 'eps': eps } , constraints=nonlinear_constraint)
    return result.fun




if __name__ == "__main__":
    ...
