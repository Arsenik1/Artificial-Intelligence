import numpy as np
import copy

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
def reward(state,new_state,F, step):
    penalty_diff = new_state[0]**2 +  0.25*new_state[1]**2 + 0.0025* new_state[2]**2 + 0.0025* new_state[3]**2
    penalty_fall = (abs(new_state[0]) >= np.pi / 2) * 1000
    balance_reward = 1 / (1 + abs(new_state[0]))  # Add balance reward based on the angle of the pendulum
    balance_duration = step * 0.3  # Add balance duration based on the number of steps
	# ..............................................
    # ..............................................
    return -(penalty_diff + penalty_fall) + balance_reward + balance_duration


def reproduction(best_population, population_size, p_mutation, p_crossover, mutation_scale=0.1):
    new_population = []
    for _ in range(population_size):
        random_row = np.random.choice(best_population.shape[0])
        parent1 = best_population[random_row]
        random_row = np.random.choice(best_population.shape[0])
        parent2 = best_population[random_row]
        child = crossover(parent1, parent2, p_crossover)
        mutated_child = mutation(child, p_mutation, mutation_scale)
        new_population.append(mutated_child)
    return new_population

def crossover(parent1, parent2, p_crossover):
    child = np.zeros(parent1.shape)
    for i in range(parent1.shape[0]):
        if np.random.rand() < p_crossover:
            child[i] = (parent1[i] + parent2[i]) / 2.0
        else:
            child[i] = parent1[i]
    return child


def mutation(individual, p_mutation, mutation_scale=0.1):
    mutated_individual = individual.copy()
    for i in range(mutated_individual.shape[0]):
        if np.random.uniform() < p_mutation:
            # Add random noise to the parameter with a scale defined by mutation_scale
            mutated_individual[i] += np.random.normal(scale=mutation_scale)
    return mutated_individual

def inv_pendulum_test(initial_states, controller, best_individual):
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
    for episode in range(num_of_initial_states):
        # Choose initial state:
        # state = rand(1,4).*[np.pi/1.5, np.pi/1.5, 20, 20] - [np.pi/3, np.pi/3, 10, 10] # random choose of initial state
        initial_state_no = episode
        state = initial_states[initial_state_no, :]

        step = 0
        reward_sum_in_episode = 0
        if_pendulum_fall = 0
        while (step < 1000) & (if_pendulum_fall == 0):
            step += 1

            # We determine actions a (forces) in the state according to the learned strategy
            # (without exploration)
            # ........................................................
            # ........................................................
            F = controller(best_individual, state)
            # F = 200   # for now

            # new state determination:
            new_state = next_state(state, F)

            if_pendulum_fall = (abs(new_state[0]) >= np.pi / 2)
            R = reward(state, new_state, F, step)
            reward_sum_in_episode += R

            pli.write(str(episode + 1) + "  " + str(state[0]) + "  " + str(state[1]) + "  " + str(state[2]) + "  " + str(state[3]) + "  " + str(F) + "\n")

            state = new_state

        avg_reward_sum = avg_reward_sum + reward_sum_in_episode / num_of_initial_states
        num_of_steps = num_of_steps + step
        print("in %d episode: sum of rewards = %g, number of steps = %d" %(episode, reward_sum_in_episode, step))

    print("average reward sum in episode = %g" % (avg_reward_sum))
    print("average number of steps of episode = %g" % (num_of_steps/num_of_initial_states))

    pli.close()
    
import torch
import torch.nn as nn

class NeuralController(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralController, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def neural_genetic_controller(model, state):
    # Convert the state to a PyTorch tensor
    state_tensor = torch.tensor(state, dtype=torch.float32)
    
    # Pass the state through the neural network to get the force
    force = model(state_tensor).item()
    
    return force



# def genetic_controller(weights, state):
#     # Calculate the force as a linear combination of the state variables and the weight vector
#     force = np.dot(weights, state)
#     return force


# framework for training of inverted pendulum system controller by reinforcement learning or
# other stage to stage algorithm:
def inv_pendulum_train_neuroevolution(num_of_episodes, population_size, selection_factor, p_cross, p_mut, mutation_scale):
    initial_states = np.array([	[np.pi/6,0, 0, 0],
								[0, np.pi/3, 0, 0], 
								[0, 0, -10, 1], 
								[0, 0, 0, -10], 
								[np.pi/12, np.pi/6, 0, 0],
								[np.pi/12, -np.pi/6, 0, 0], 
								[-np.pi/12, np.pi/6, 0, 0], 
								[-np.pi/12, -np.pi/6, 0, 0],
								[np.pi/12, 0, 0, 0], 
								[0, 0, -10, 10]], dtype=float)
	
	# Create the initial population of NeuralController models
    population = [NeuralController(input_size=4, hidden_size=10, output_size=1) for _ in range(population_size)]
    
    for episode in range(num_of_episodes):
        # Evaluate the performance of each model in the population
        rewards = []
        for model in population:
            reward_sum_in_episode = 0
            for initial_state in initial_states:
                state = initial_state.copy()
                step = 0
                if_pendulum_fall = 0
                while (step < 300) & (if_pendulum_fall == 0):
                    step += 1

                    F = neural_genetic_controller(model, state)
                    new_state = next_state(state, F)
                    if_pendulum_fall = (abs(new_state[0]) >= np.pi / 2)
                    R = reward(state, new_state, F, step)
                    reward_sum_in_episode += R
                    state = new_state
            rewards.append(reward_sum_in_episode)
        
        # Select the best performing models
        rewarded_solutions = list(zip(rewards, population))
        rewarded_solutions.sort(key=lambda x: x[0], reverse=True)
        best_solutions = rewarded_solutions[:int(selection_factor * population_size)]
        best_population = [x[1] for x in best_solutions]
        
        # Create a new population by reproducing the best models
        new_population = []
        for _ in range(population_size):
            parent1 = np.random.choice(best_population)
            parent2 = np.random.choice(best_population)
            child = crossover_neuroevolution(parent1, parent2, p_cross)
            mutated_child = mutation_neuroevolution(child, p_mut, mutation_scale)
            new_population.append(mutated_child)
        population = new_population
    
    # Return the best performing model from the final population
    best_model = max(zip(rewards, population), key=lambda x: x[0])[1]
    return best_model

def crossover_neuroevolution(parent1, parent2, p_crossover):
    child = copy.deepcopy(parent1)
    for name, param in parent1.named_parameters():
        if np.random.rand() < p_crossover:
            param.data.copy_((param.data + parent2.state_dict()[name].data) / 2.0)
    return child

def mutation_neuroevolution(model, p_mutation, mutation_scale=0.1):
    mutated_model = copy.deepcopy(model)
    for param in mutated_model.parameters():
        if np.random.uniform() < p_mutation:
            noise = torch.randn_like(param) * mutation_scale
            param.data.add_(noise)
    return mutated_model


inv_pendulum_train_neuroevolution(num_of_episodes=1000, population_size=300, selection_factor=0.6, p_cross=0.8, p_mut=0.08, mutation_scale=0.4)


