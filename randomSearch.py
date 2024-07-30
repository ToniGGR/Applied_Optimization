from cyclone import *

def objective (x):
    #print(fun_cyclone(x, model="Barth-Muschelknautz")[:2])
    return fun_cyclone(x, model="Barth-Muschelknautz")[:2]

def random_search(n=6,fun=objective , lower=[1,2,0.3,0.5,0.5,0.1] , 
                  upper=[1.5,3,0.5,0.8,0.7,0.3], budget=1000, upper_pl=1000):   # budget gibt an wie viele Arrays erstellt werden (mit random x Werten)
    lower, upper = np.array(lower), np.array(upper)
    x = np.random.uniform(lower,upper, (budget, n))
    results = np.array([fun(xi)for xi in x]) # results ist fuer alle 1000 Kombinationen die Pl und E Werte
    feasible = results[:,0] <= upper_pl #feasible ist ein Array wo zuruekcgegebwn wird wann pl<1000 erfuellt ist
    feasible_results = results[feasible] #feasible_results gibt fuer alle feasible Kombinationen die E und Pl Werte an
        

    if feasible_results.size == 0: # keine Kombination erfuellt die Bedingung
        return None
    
    id_best_feasible = np.argmin(feasible_results[:, 1]) #Hole die ID mit dem geringensten E Wert , da so nur Minimum erkannt werden muss
    best_feasible = feasible_results[id_best_feasible]

    ids = np.all(results == best_feasible, axis=1)

    return {
        'Ebest' : best_feasible[1],
        'PL' : best_feasible[0],
        'xbest' : x[ids]
    }

def test_rs():
    optim_result = random_search()
    x_array_meaning = ["Da", "H" , "Dt" , "Ht" , "He" , "Be"]
    result_array_meaning = ["Pressure Lost" , "Efficiency" , "Weighted Efficiency"]
    print("Optimization Result E:" , optim_result['Ebest'])
    print("Optimization Result PL:" , optim_result['PL'])
    print("Optimization Result X:" , optim_result['xbest'])
    print(f"Meaning of x_array \t\t {x_array_meaning}")
    print(f"Meaning of result_array \t\t {result_array_meaning}")

    #print("Check Objective on xbest:" , objective(optim_result['xbest']))

    

test_rs()