from cyclone import calculation_barth_muschelknautz
import numpy as np
from neldermead import nedermead_perform
from randomSearch import test_rs
from cobyla import *

def main():
    cyclone = {'Da': 2.0, 'Dt': 1.0, 'H': 5.0, 'Ht': 3.0, 'He': 1.0, 'Be': 0.5}
    fluid = {'Vp': 1.2, 'Croh': 0.05, 'Rhof': 1, 'Rhop': 1.2, 'Mu': 0.01, 'lambdag': 0.02}
    xmean = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    delta = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    efficiency, ew, pressure_drop = calculation_barth_muschelknautz(cyclone, fluid, xmean, delta)
    print("Efficiency:", efficiency)
    print("Ew (weighted efficiency):", ew)
    print("Pressure Drop:", pressure_drop)


def test():
    best_x = [0,0,0,0,0,0]
    max_predict = 0
    for i in range(50):
        i = np.random.uniform([1,2,0.3,0.5,0.5,0.1] , [1.5,3,0.5,0.8,0.7,0.3])
        try:
            res = cobyla_perform(initial_guess= i)
            if max_predict > res["E"] and res["PL"] < 1000:
                max_predict = res["E"]
                best_x = res["X"]
                best_P = res["PL"]
        except:
            continue

    print(max_predict , best_x , best_P)

def compare():
    nedermead_perform()
    print(f"\n#########################################################\n")
    test_rs()


if __name__ == "__main__":
    test()