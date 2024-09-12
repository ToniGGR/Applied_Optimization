import numpy as np
import optuna
from scipy.optimize import minimize
from cyclone import fun_cyclone

def objective (x , penalty=0.85):
    ### WEIGHTED PENALTY
    if constraint(x) > 1000:
        weight = constraint(x) - 1000
        return fun_cyclone(x, model="Barth-Muschelknautz")[:2][1] *  - (penalty * 0.001 * weight)
    return fun_cyclone(x, model="Barth-Muschelknautz")[:2][1]


#Pressure Loss Constraint
def constraint (x):
    return fun_cyclone(x, model="Barth-Muschelknautz")[:2][0]


# Optuna-Zielfunktion zur Hyperparameteroptimierung
def optuna_objective(trial):
    # Hyperparameter to be tuned
    max_iter = trial.suggest_int('maxiter', 50, 500)
    

    # Start of Optimization
    x0 = np.array([1.34146605 , 2.9433389 ,  0.40951961, 0.56850335 , 0.50914978 , 0.1186709 ])

    # Minimize Function
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': max_iter})

    return result.fun

# start Optuna-Study
study = optuna.create_study(direction='minimize')
study.optimize(optuna_objective, n_trials=100)

# Beste gefundene Parameter und Ergebnis
best_trial = study.best_trial
print(f"Beste Parameter: {best_trial.params}")
print(f"Beste Zielfunktion: {best_trial.value}")
