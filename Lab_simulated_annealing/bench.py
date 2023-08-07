import map_min_search as mm
import numpy as np

num_of_steps = 2000
num_of_parameters = 2
N = num_of_parameters

T = 1.0
T_min = 1e-3
wT = 0.999
c = 1.0

Solution = np.random.rand(N) * 20 - 10

E_min = 10e40
E_prev = 0
Records = np.empty((0, N))

mm.show_the_point(Solution, "initial solution")

for ep in range(num_of_steps):
    SolutionNew = Solution + np.random.randn(N)

    E = mm.fun3(SolutionNew[0], SolutionNew[1])

    dE = E - E_prev

    p_accept = np.exp(-dE / (c * T)) #simulated annealing formula
    if np.random.rand() < p_accept:
        Solution = SolutionNew
        E_prev = E

    if E_min > E:
        print("new minimum = " + str(E) + " for point x1 = " + str(SolutionNew[0]) + " x2 = " + str(SolutionNew[1]) + "\n")
        E_min = E
        Solution_min = SolutionNew
        Records = np.append(Records, [SolutionNew], axis=0)

    T *= wT
    if T < T_min:
        T = T_min

text = "best solution, value = " + str(E_min) + " for point x1 = " + str(Solution_min[0]) + " x2 = " + str(Solution_min[1])
print(text + "\n")
mm.show_point_sequence(Records, "record sequence, " + text)
