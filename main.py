from cyclone import calculation_barth_muschelknautz
from cyclone import fun_cyclone
import numpy as np
from neldermead import nedermead_perform
from randomSearch import test_rs , random_search
from cobyla import *
from pyfiglet import Figlet
from neldermead import nedermead_perform
from cobyla import cobyla_perform, cobyqa_perform, slsqp_perform
from direct import direct_perform
from cma_es import cma_execute
from benchmark import *

def main():
    #Start Text
    f = Figlet(font="slant")
    print(f.renderText("Welcome to Applied Optimization Techniques"))
    stop = False

    #Input Loop
    while not stop:

        print("\n\n\nPress 1 for Handing Over your own Trial \n" +
            "Press 2 for Starting Algorithm \n" +
            "Press 3 for Starting Benchmarking \n" +
            "Press q to quit the program \n")

        action = input()

        # Check User Input
        match action:
            case "1":
                input_x = [0] * 6
                for _ in range(len(input_x)):
                    input_x[_] = float(input(f"Whats x{_} ?"))
                try:
                    print(fun_cyclone(input_x))
                except TypeError:
                    print("Input was not valid")
                finally:
                    continue

            case "2":
                print("Which Algorithm do you want to perform?" + 
                      "\t Press RS for Start Random Search \n" +
                       "\t Press NM for Start Nelder Mead \n" +
                        "\t Press CL for Start COBYLA \n" + 
                        "\t Press CQ for Start COBYQA \n" + 
                        "\t Press SL for Start SLSQP \n" + 
                        "\t Press DI for Start DIRECT \n" + 
                        "\t Press CM for Start CMA_ES \n")
                alg_choice = input("Input : ")

                match alg_choice:
                    case "RS":
                        print(random_search())
                    case "NM":
                        print(nedermead_perform())
                    case "CL":
                        print(cobyla_perform())
                    case "CQ":
                        print(cobyqa_perform())
                    case "SL":
                        print(slsqp_perform())
                    case "DI":
                        print(direct_perform())
                    case "CM":
                        print(cma_execute())

            #Benchmark
            case "3":
                print("Which Benchmark do you want to start? \n" +
                      "1: Amount Cyclone Functions \n" + 
                       "2: Execution Time \n" +
                        "3: Validity \n" +
                         "4: Performance \n" +
                          "5: Loss Functions" )
                bm_input = input("Input: ")

                match bm_input:
                    case "1":
                        compare_iterations()
                    case "2":
                        compare_execution_time()
                    case "3":
                        compare_Validiy()
                    case "4":
                        compare_scatter()
                    case "5":
                        compare_progress()

            # Stop Program
            case "q":
                print(f.renderText("Stopped"))
                stop = True

# Main Function
if __name__ == "__main__":
    main()