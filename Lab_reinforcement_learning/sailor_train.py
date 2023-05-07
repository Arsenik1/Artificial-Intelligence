import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf

number_of_episodes = 4000                   # number of training episodes (multi-stage processes) 
gamma = 1.0                                 # discount factor

alpha = 0.05                                # training speed factor
epsilon = 0.01                               # exploration factor

#file_name = 'map_small.txt'
file_name = 'D:\Desktop\Salih\Belgeler\VSCCode\Python\AI\Lab_rf2023_ver1\map_small.txt'
#file_name = 'D:\Desktop\Salih\Belgeler\VSCCode\Python\AI\Lab_rf2023_ver1\map_middle.txt'
#file_name = 'map_big.txt'
#file_name = 'map_spiral.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(2.5*(num_of_rows + num_of_columns))    # maximum number of steps in an episode
Q = np.zeros([num_of_rows, num_of_columns, 4], dtype=float)  # trained usability table of <state,action> pairs
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

for episode in range(number_of_episodes):
    state = np.random.randint(low=0, high=num_of_rows, size=2)   # initial state is random due to exploration

    the_end = False
    nr_pos = 0
    while the_end == False:
        nr_pos = nr_pos + 1                            # move number
      
        # Action choosing (1 - right, 2 - up, 3 - left, 4 - bottom): 
        action = sf.choose_action(state, Q, epsilon)   # choose action with epsilon-greedy policy

        state_next, reward = sf.environment(state, action, reward_map)
      
        # State-action usability modification:
        Q[state[0], state[1], action-1] += alpha * (reward + gamma * np.max(Q[state_next[0], state_next[1], :]) - Q[state[0], state[1], action-1])
        
        state = state_next      # going to the next state
      
        # end of episode if maximum number of steps is reached or last column is reached
        if (nr_pos == num_of_steps_max) or (state[1] >= num_of_columns-1):
            the_end = True                                  
      
        sum_of_rewards[episode] += reward
    if episode % 500 == 0:
        print('episode = ' + str(episode) + ' average sum of rewards = ' + str(np.mean(sum_of_rewards)))
print('average sum of rewards = ' + str(np.mean(sum_of_rewards)))

sf.sailor_test(reward_map, Q, 1000)
sf.draw(reward_map, Q)
