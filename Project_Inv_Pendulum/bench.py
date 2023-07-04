import numpy as np
import random as random

def global_variables():
    Fmax = 1200
    time_step = 0.05     # time between two subsequent calculated states  
    g = 9.8135           # the vertical acceleration
    friction = 0.02
    cart_weight = 20     
    pend_weight = 20
    drw = 25             # the length of pendulum
    return Fmax, time_step, g, friction, cart_weight, pend_weight, drw


# Calculation of the state of the pendulum in the next time step by analytical method
# state - vector of state parameters in time t
# state_n -   -||- in time t + time_step
# F - force applied to the cart
def next_state(state,F):
    Fmax, time_step, g, friction, cart_weight, pend_weight, drw = global_variables()

    if F>Fmax:
        F=Fmax
    if F<-Fmax:
        F=-Fmax

    hh = time_step * 0.5
    cart_mom = cart_weight * drw
    pend_mom = pend_weight * drw
    cwoz = cart_weight * g
    cwah = pend_weight * g

    sx=np.sin(state[0])
    cx=np.cos(state[0])
    c1=cart_weight+pend_weight*sx*sx
    c2=pend_mom*state[1]*state[1]*sx
    c3=friction*state[3]*cx

    stanpoch = np.zeros(state.size)

    stanpoch[0]=state[1]
    stanpoch[1]=((cwah+cwoz)*sx-c2*cx+c3-F*cx)/(drw*c1)
    stanpoch[2]=state[3]
    stanpoch[3]=(c2-cwah*sx*cx-c3+F)/c1
    stanh = np.zeros(state.size)
    for i in range(4):
        stanh[i]=state[i]+stanpoch[i]*hh
  
    sx=np.sin(stanh[0])
    cx=np.cos(stanh[0])
    c1=cart_weight+pend_weight*sx*sx
    c2=pend_mom*stanh[1]*stanh[1]*sx
    c3=friction*stanh[3]*cx

    stanpochh = np.zeros(state.size)
    stanpochh[0]=stanh[1]
    stanpochh[1]=((cwah+cwoz)*sx-c2*cx+c3-F*cx)/(drw*c1)
    stanpochh[2]=stanh[3]
    stanpochh[3]=(c2-cwah*sx*cx-c3+F)/c1
    state_n = np.zeros(state.size)
    for i in range(4):
        state_n[i]=state[i]+stanpochh[i]*time_step
    if state_n[0] > np.pi:
        state_n[0]=state_n[0]-2*np.pi
    if state_n[0] < -np.pi:
        state_n[0]=state_n[0]+2*np.pi

    return state_n

# reward for transition from state to new state with action F 
def reward(state,new_state,F):
    penalty_diff = new_state[0]**2 +  0.25*new_state[1]**2 + 0.0025* new_state[2]**2 + 0.0025* new_state[3]**2
    penalty_fall = (abs(new_state[0]) >= np.pi / 2) * 1000

    return -(penalty_diff + penalty_fall)

# 
def inv_pendulum_test(initial_states, controller):
    Fmax, time_step, g, friction, cart_weight, pend_weight, drw = global_variables()
    pli = open('history.txt', 'w')
    pli.write("Fmax = " + str(Fmax) + "\n")
    pli.write("time_step = " + str(time_step) + "\n")
    pli.write("g = " + str(g) + "\n")
    pli.write("friction = " + str(friction) + "\n")
    pli.write("cart_weight = " + str(cart_weight) + "\n")
    pli.write("pend_weight = " + str(pend_weight) + "\n")
    pli.write("drw = " + str(drw) + "\n")

    avg_reward_sum = 0
    num_of_steps = 0
    num_of_initial_states, lparam = initial_states.shape

    p_cross = 0.62
    p_mut = 0.19
    selection_factor = 1.0


    population_size = 50
    num_of_weights = 4       # for now
    population = np.random.uniform(low=-1, high=1, size=(population_size, num_of_weights))

    for episode in range(num_of_initial_states):
        # Choose initial state:
        # state = rand(1,4).*[np.pi/1.5, np.pi/1.5, 20, 20] - [np.pi/3, np.pi/3, 10, 10] # random choose of initial state
        initial_state_no = episode
        state = initial_states[initial_state_no, :]

        step = 0
        reward_sum_in_episode = 0
        if_pendulum_fall = 0
        ranked_population = []
        while (step < 1000) & (if_pendulum_fall == 0):
            step += 1

            # We determine actions a (forces) in the state according to the learned strategy
            # (without exploration)
            # ........................................................
            # ........................................................
            individual_index = np.random.randint(population_size)
            individual = population[individual_index]
            F = controller(individual,state)
            # F = controller(state)
            # F = 200   # for now

            # new state determination:
            new_state = next_state(state, F)

            if_pendulum_fall = (abs(new_state[0]) >= np.pi / 2)
            R = reward(state, new_state, F)
            reward_sum_in_episode += R

            pli.write(str(episode + 1) + "  " + str(state[0]) + "  " + str(state[1]) + "  " + str(state[2]) + "  " + str(state[3]) + "  " + str(F) + "\n")

            ranked_population.append((individual_index, R));        
            state = new_state

        ranked_population.sort(key=lambda x: x[1], reverse=True)

        best_population = ranked_population[:int(population_size*0.1)]

        rewards =  [x[1] for x in ranked_population]
        new_population = reproduction(best_population, rewards[0:int(population_size*selection_factor)], population_size)
        mutated_population = mutation(new_population, p_mut) 
        crossovered_population = crossover(mutated_population, p_cross)

        population = crossovered_population

        avg_reward_sum = avg_reward_sum + reward_sum_in_episode / num_of_initial_states
        num_of_steps = num_of_steps + step
        print("in %d episode: sum of rewards = %g, number of steps = %d" %(episode, reward_sum_in_episode, step))

    print("average reward sum in episode = %g" % (avg_reward_sum))
    print("average number of steps of episode = %g" % (num_of_steps/num_of_initial_states))

    pli.close()

def genetic_controller(individual, state):
    # Generate a random force
    Fmax, time_step, g, friction, cart_weight, pend_weight, drw = global_variables()

    force = np.random.uniform(-Fmax, Fmax)
    return force

# def genetic_controller(individual, state):
#     # print(f"Shape of individual: {individual.shape}")
#     # Use a neural network with weights determined by the individual to calculate the force
#     Fmax, time_step, g, friction, cart_weight, pend_weight, drw = global_variables()
    
#     # Reshape the individual into a matrix of weights for the neural network
#     weights = individual.reshape((4, 1))
    
#     # Calculate the force as the dot product of the state and the weights
#     force = np.dot(state, weights)
    
#     # Clip the force to be within the range of [-Fmax, Fmax]
#     force = np.clip(force, -Fmax, Fmax)
    
#     return force



def crossover(population, p_cross):
    new_population = population.copy()
    for i in range(population.shape[0]-1):
        if np.random.rand() < p_cross:
            rand_index = np.random.randint(population.shape[0])
            new_population[i] = (population[i] + population[rand_index])/2.0
    return new_population


def mutation(population, p_mut):
    mutated_population = []
    for individual in population:
        if np.random.uniform() < p_mut:
            mutated_individual = individual + np.random.normal(loc=0, scale=0.1)
            mutated_population.append(mutated_individual)
        else:
            mutated_population.append(individual)
    return np.array(mutated_population)


def reproduction(best_population, rewards, population_size):
    new_population = []
    for i in range(population_size):
        parent1 = best_population[np.random.randint(len(best_population))][0]
        parent2 = best_population[np.random.randint(len(best_population))][0]
        child = (parent1 + parent2) / 2
        new_population.append(child)
    return new_population


# framework for training of inverted pendulum system controller by reinforcement learning or
# other stage to stage algorithm:
def inv_pendulum_train(num_of_episodes):

    initial_states = np.array([ [np.pi/6,0, 0, 0],
                                [0, np.pi/3, 0, 0],
                                [0, 0, -10, 1],
                                [0, 0, 0, -10],
                                [np.pi/12, np.pi/6, 0, 0],
                                [np.pi/12, -np.pi/6, 0, 0],
                                [-np.pi/12, np.pi/6, 0, 0],
                                [-np.pi/12, -np.pi/6, 0, 0],
                                [np.pi/12, 0, 0, 0],
                                [0, 0, -10, 10]], dtype=float)
    
    # The four state variables, in order, are:

    # initial_states[:, 0] - Angle of the pendulum (theta)
    # initial_states[:, 1] - Angular velocity of the pendulum (theta_dot)
    # initial_states[:, 2] - Position of the cart (x)
    # initial_states[:, 3] - Velocity of the cart (x_dot)

    num_of_initial_states, lparam = initial_states.shape

    # initiation of coding, determination of the number of parameters (weights):
    num_of_parameters = 2
    num_of_epochs = 50
    
    p_cross = 0.62
    p_mut = 0.19
    selection_factor = 1.0
    controller = genetic_controller  # for now


    # controller parameter vector initialization:
    population_size = 50
    num_of_weights = 4       # for now
    w = np.zeros(num_of_weights)
    population = np.random.uniform(low=-1, high=1, size=(population_size, num_of_weights))

    for episode in range(num_of_episodes):
        # Choose initial state:
        # state = np.multiply(
        #         [np.pi / 1.5, np.pi / 1.5, 20, 20],
        #         [random.uniform(0, 1), random.uniform(0, 1), 
        #             random.uniform(0, 1), random.uniform(0, 1)]) - [np.pi / 3, np.pi / 3, 10, 10]
        # state = rand(1,4).*[np.pi/1.5, np.pi/1.5, 20, 20] - [np.pi/3, np.pi/3, 10, 10] # random choose of initial state
        initial_state_no = episode %  num_of_initial_states
        state = initial_states[initial_state_no, :]

        step = 0
        if_pendulum_fall = 0
        ranked_population = []
        while (step < 1000) & (if_pendulum_fall == 0):
            step += 1
            
            individual_index = np.random.randint(population_size)
            individual = population[individual_index]
            F = controller(individual,state)
            # F = 200  # for now

            # new state determination:
            new_state = next_state(state, F)

            if_pendulum_fall = (abs(new_state[0]) >= np.pi / 2)
            R = reward(state, new_state, F)

            ranked_population.append((individual_index, R));        
            state = new_state

        ranked_population.sort(key=lambda x: x[1], reverse=True)

        best_population = ranked_population[:int(population_size*0.1)]

        rewards =  [x[1] for x in ranked_population]
        new_population = reproduction(best_population, rewards[0:int(population_size*selection_factor)], population_size)
        mutated_population = mutation(new_population, p_mut) 
        crossovered_population = crossover(mutated_population, p_cross)

        population = crossovered_population

        # test + history file generation:
        if episode % 1000 == 0:
            inv_pendulum_test(initial_states, controller)


inv_pendulum_train(num_of_episodes = 20)


