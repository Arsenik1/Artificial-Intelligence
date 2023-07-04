import numpy as np
import tensorflow as tf

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
    penalty_diff = abs(new_state[0])**2 +  0.25*(abs(new_state[1]))**2 + 0.0025* abs(new_state[2])**2 + 0.0025* abs(new_state[3])**2
    penalty_fall = (abs(new_state[0]) >= np.pi / 2) * (200 / step) * 1000
	# ..............................................
    # ..............................................
    return -(penalty_diff + penalty_fall)


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
	index = 0
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
			F = neural_controller(model, preprocess_state(state))
			print("\n-------------------------------------\nF = ", F)
			# F = 200   # for now

			# new state determination:
			new_state = next_state(state, F)
			print("\ntheta = ", new_state[0], "\ntheta_dot = ", new_state[1], "\nx = ", new_state[2], "\nx_dot = ", new_state[3])

			if_pendulum_fall = (abs(new_state[0]) >= np.pi / 2)
			R = reward(state, new_state, F, step)
			reward_sum_in_episode += R

			pli.write(str(episode + 1) + "  " + str(state[0]) + "  " + str(state[1]) + "  " + str(state[2]) + "  " + str(state[3]) + "  " + str(F) + "\n")

			state = new_state

		if if_pendulum_fall == 0:
			print("\nPendulum did not fall in episode %d\n" % (episode))

		avg_reward_sum = avg_reward_sum + reward_sum_in_episode / num_of_initial_states
		num_of_steps = num_of_steps + step
		print("in %d episode: sum of rewards = %g, number of steps = %d" %(episode, reward_sum_in_episode, step))

	print("average reward sum in episode = %g" % (avg_reward_sum))
	print("average number of steps of episode = %g" % (num_of_steps/num_of_initial_states))
	print(best_individual)
	pli.close()

# def genetic_controller(weights, state):
# 	theta, theta_dot, x, x_dot = state
# 	Fmax, time_step, g, friction, cart_weight, pend_weight, drw = global_variables()


# 	return force      	

# def genetic_controller(weights, state):
#     # Extract the state variables
#     theta, theta_dot, x, x_dot = state

#     # Define the system parameters
#     Fmax, time_step, g, friction, cart_weight, pend_weight, drw = global_variables()
#     m = pend_weight  # Mass of the pendulum
#     M = cart_weight  # Mass of the cart
#     l = drw  # Length of the pendulum

#     # Calculate the state derivatives using the equations of motion
#     x_ddot = (m * l * theta_dot**2 * np.sin(theta) - m * g * np.sin(theta) * np.cos(theta)) / (M + m - m * np.cos(theta)**2)
#     theta_ddot = (g * np.sin(theta) - np.cos(theta) * x_ddot) / l

#     # Calculate the force using a non-linear combination of the state variables and the weight vector
#     force = weights[0] * x_ddot + weights[1] * theta_ddot + weights[2] * np.sin(theta) + weights[3] * np.sin(x)
    
#     return force


# def genetic_controller(w, state):
	
# 	# theta, theta_dot, x, x_dot = state
# 	# bias = w[4]

# 	# force = w[0] * theta + w[1] * theta_dot + w[2] * x + w[3] * x_dot + bias

# 	# Extract the state variables
# 	theta, theta_dot, x, x_dot = state

# 	# Define the system parameters
# 	m = 20.0  # Mass of the pendulum
# 	M = 20.0  # Mass of the cart
# 	l = 25.0  # Length of the pendulum
# 	g = 9.8135  # Gravitational acceleration

# 	# Calculate the state derivatives using the equations of motion
# 	x_ddot = (m * l * theta_dot**2 * np.sin(theta) - m * g * np.sin(theta) * np.cos(theta)) / (M + m - m * np.cos(theta)**2)
# 	theta_ddot = (g * np.sin(theta) - np.cos(theta) * x_ddot) / l

# 	# Calculate the force as a linear combination of the state derivatives and the weight vector
# 	force = w[0] * x_ddot + w[1] * theta_ddot
# 	return force


# def genetic_controller(weights, state):
#     # Calculate the force as a linear combination of the state variables and the weight vector
#     force = np.dot(weights, state)
#     return force
        
input_size = 4  # Number of state variables
output_size = 1  # Number of force outputs

model = tf.keras.Sequential([
tf.keras.layers.Dense(64, activation='relu', input_shape=(input_size,)),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(output_size)
])
def preprocess_state(state):
    return state.reshape((1, input_size))

def neural_controller(model, preprocessed_state):
    force = model.predict(preprocessed_state)
    return force

# framework for training of inverted pendulum system controller by reinforcement learning or
# other stage to stage algorithm:
def inv_pendulum_train(num_of_episodes):
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
	num_of_initial_states, lparam = initial_states.shape
	# The four state variables, in order, are:
	# chosen_solutions = []
	# chosen_solutions = np.array(chosen_solutions)
	# initial_states[:, 0] - Angle of the pendulum (theta)
	# initial_states[:, 1] - Angular velocity of the pendulum (theta_dot)
	# initial_states[:, 2] - Position of the cart (x)
	# initial_states[:, 3] - Velocity of the cart (x_dot)

	# initiation of coding, determination of the number of parameters (weights):
	p_cross = 0.6
	p_mut = 0.8
	mutation_scale=0.9
	selection_factor = 0.1
	controller = genetic_controller
        
	

	# controller parameter vector initialization:
	num_of_weights = 4
	population_size = 1000
	w = np.random.uniform(low=-1, high=1, size=(population_size, num_of_weights))
	# population = np.random.uniform(low=-1, high=1, size=(population_size, num_of_weights))
	# population[:, 0] = np.random.uniform(low=-np.pi/3, high=np.pi/3, size=population_size)
	# population[:, 1] = np.random.uniform(low=-np.pi/3, high=np.pi/3, size=population_size)
	for episode in range(num_of_episodes):
		# Choose initial state:
		# state = np.random.rand(4) * np.array([np.pi/1.5, np.pi/1.5, 20, 20]) - np.array([np.pi/3, np.pi/3, 10, 10])

		# state = rand(1,4).*[np.pi/1.5, np.pi/1.5, 20, 20] - [np.pi/3, np.pi/3, 10, 10] # random choose of initial state
		initial_state_no = episode %  num_of_initial_states
		state = initial_states[initial_state_no, :]

		step = 0
		if_pendulum_fall = 0
		rewards = []
		rewarded_solutions = []
		for weight in w:
			state = initial_states[initial_state_no, :]
			while (step < 1000) & (if_pendulum_fall == 0):
				step += 1
				
				F = controller(weight, state)
				# F = 200  # for now

				# new state determination:
				new_state = next_state(state, F)

				if_pendulum_fall = (abs(new_state[0]) >= np.pi / 2)
				state = new_state
			R = reward(state, new_state, F, step)
			rewards.append(R)
			rewarded_solutions.append((R, weight, state))


		rewarded_solutions.sort(key=lambda x: x[0], reverse=True)

		best_solutions = rewarded_solutions[:int(selection_factor * population_size)]
		best_solutions = [x[1] for x in best_solutions]
		best_solutions = np.array(best_solutions)

		#reproduction:
		w = reproduction(best_solutions, population_size, p_cross, p_mut, mutation_scale)

		# test + history file generation:
		if episode % (num_of_episodes / 5) == 0:
			# print("\n----------------------------------\n")
			# print(rewarded_solutions[:][0])
			# print("\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")
			# print(best_solutions[0])
			# print("\n----------------------------------\n")
			inv_pendulum_test(initial_states, controller, best_solutions[0])

	inv_pendulum_test(initial_states, controller, best_solutions[0])

np.random.seed(0)
inv_pendulum_train(num_of_episodes = 60)


