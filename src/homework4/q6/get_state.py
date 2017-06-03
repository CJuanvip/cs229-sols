import numpy as np


def get_state(x, x_dot, theta, theta_dot):
    """
    This function returns a discretized value (a number) for a continuous
    state vector. Currently x is divided into 3 "boxes", x_dot into 3,
    theta into 6 and theta_dot into 3. A finer discretization produces a
    larger state space, but allows a better policy.
    """

    # Parameters for state discretization in get_state
    one_degree     = 0.0174532 # 2pi/360
    six_degrees    = 0.1047192
    twelve_degrees = 0.2094384
    fifty_degrees  = 0.87266

    total_states = 163

    state = 0

    if (x < -2.4) or (x > 2.4) or (theta < -twelve_degrees) or (theta > twelve_degrees):
        state = total_states - 1 # to signal failure
    else:
        # Check the state of x.
        if x < -1.5:
            state = 0
        elif x < 1.5:
            state = 1
        else:        
            state = 2
  
        # Check the state of x_dot.
        if x_dot < -0.5:
            state = state # No change
        elif x_dot < 0.5:
            state = state + 3
        else:
            state = state + 6
     
        # Check the state of theta.
        if theta < -six_degrees:  
            state = state # No change
        elif theta < -one_degree:
            state = state + 9
        elif theta < 0:
            state = state + 18
        elif theta < one_degree:
            state = state + 27
        elif theta < six_degrees:
            state = state + 36
        else:
            state = state + 45
        
        # Check the state of theta_dot.
        if theta_dot < -fifty_degrees:
            state = state # No change
        elif theta_dot < fifty_degrees:  
            state = state + 54
        else:
            state = state + 108  

    # This is because of MATLAB indexing.
    # state = state + 1

    return state