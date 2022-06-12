import numpy as np


def action_to_vector(action):
    return np.concatenate((action.position, action.torque))


def vector_to_action(vector, sim):
    n = vector.shape[0] // 2
    pos, torque = vector[:n], vector[n:]
    return sim.Action(position=pos, torque=torque)


def get_vector_state(finger_state, cube_state, goal_state):
    _, finger_state_ = zip(*finger_state.__dict__.items())
    finger_state_ = np.concatenate(finger_state_)
    cube_state_ = cube_state[0]
    return np.concatenate((finger_state_, cube_state_, goal_state))
