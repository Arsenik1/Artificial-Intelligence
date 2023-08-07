"""
   Simulated Annealing  - the minimum of 2D function searching

"""
import map_min_search as mm
import numpy as np
                                  
num_of_steps = 2000                                 # number of steps: do not change

num_of_parameters = 2                               # number of solution parameters
N = num_of_parameters

T = ???                                             # temperature (randomness coefficient)
T_min = ???                                         # minimal temperature
wT = ???                                            # change of temperature
c = ???                                             # constant due to the influence of T for acceptance probability

Solution = np.random.rand(N)*20-10                  # initial solution - random point


E_min = 10e40							            # minimal function value
E_prev = 0                                          # previous value of the function
Records = np.empty((0,N))                           # array of record solutions

mm.show_the_point(Solution,"initial solution")

for ep in range(num_of_steps):
   SolutionNew = ???                                # new solution (should be near previous one !)

   E = mm.fun3(SolutionNew[0],SolutionNew[1])       # function value for point coordinates

   dE = E - E_prev                                  # change of function value (dE < 0 means than new solution is better)

   p_accept = ???                                   # acceptance probability
   if np.random.rand() < p_accept:
      Solution = SolutionNew
      E_prev = E

   if E_min > E:
      print("new minimum = " + str(E) + " for point x1 = " + str(SolutionNew[0]) + " x2 = " + str(SolutionNew[1]) + "\n")
      E_min = E
      Solution_min = SolutionNew
      Records = np.append(Records, [SolutionNew], axis = 0)

   T = ???                                          # temperature changing (can be only after accaptance or in another place)
   if T < T_min:
      T = T_min
# end of steps loop
text = "best solution, value = " + str(E_min) + " for point x1 = " + str(Solution_min[0]) + " x2 = " + str(Solution_min[1])
print(text + "\n")
mm.show_point_sequence(Records,"record sequence, " + text)

