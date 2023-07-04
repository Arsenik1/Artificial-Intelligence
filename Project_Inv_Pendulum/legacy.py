import numpy as np

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
    # ..............................................
    # ..............................................
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
            # F = controller(state)
            F = 0   # for now

            # new state determination:
            new_state = next_state(state, F)

            if_pendulum_fall = (abs(new_state[0]) >= np.pi / 2)
            R = reward(state, new_state, F)
            reward_sum_in_episode += R

            pli.write(str(episode + 1) + "  " + str(state[0]) + "  " + str(state[1]) + "  " + str(state[2]) + "  " + str(state[3]) + "  " + str(F) + "\n")

            state = new_state

        avg_reward_sum = avg_reward_sum + reward_sum_in_episode / num_of_initial_states
        num_of_steps = num_of_steps + step
        print("in %d episode: sum of rewards = %g, number of steps = %d" %(episode, reward_sum_in_episode, step))

    print("average reward sum in episode = %g" % (avg_reward_sum))
    print("average number of steps of episode = %g" % (num_of_steps/num_of_initial_states))

    pli.close()

# framework for training of inverted pendulum system controller by reinforcement learning or
# other stage to stage algorithm:
def inv_pendulum_train(num_of_episodes):
    
    alpha = 0.001            # training speed factor
    epsilon = 0.1            # exploration factor

    initial_states = np.array([[np.pi/6,0, 0, 0],[0, np.pi/3, 0, 0], [0, 0, -10, 1], [0, 0, 0, -10], [np.pi/12, np.pi/6, 0, 0],
                      [np.pi/12, -np.pi/6, 0, 0], [-np.pi/12, np.pi/6, 0, 0], [-np.pi/12, -np.pi/6, 0, 0],
                      [np.pi/12, 0, 0, 0], [0, 0, -10, 10]],dtype=float)
    num_of_initial_states, lparam = initial_states.shape

    # initiation of coding, determination of the number of parameters (weights):
    # ........................................................
    # ........................................................
    controller = None  # for now

    # controller parameter vector initialization:
    num_of_weights = 100       # for now
    w = np.zeros(num_of_weights)

    for episode in range(num_of_episodes):
        # Choose initial state:
        # state = rand(1,4).*[np.pi/1.5, np.pi/1.5, 20, 20] - [np.pi/3, np.pi/3, 10, 10] # random choose of initial state
        initial_state_no = episode %  num_of_initial_states
        state = initial_states[initial_state_no, :]

        step = 0
        if_pendulum_fall = 0
        while (step < 1000) & (if_pendulum_fall == 0):
            step += 1

            # We determine actions a (forces) in the state state taking into account
            # exploration (e.g. epsilon-greedy or softmax method)
            # ........................................................
            # ........................................................
            # F = controller(w,state)
            F = 0  # for now

            # new state determination:
            new_state = next_state(state, F)

            if_pendulum_fall = (abs(new_state[0]) >= np.pi / 2)
            R = reward(state, new_state, F)

            # We update the Q values for the current state and execution of the action:
            # ........................................................
            # ........................................................
            # w = w + ...

            state = new_state

        # test + history file generation:
        if episode % 200 == 0:
            inv_pendulum_test(initial_states, controller)


inv_pendulum_train(num_of_episodes = 1000)


