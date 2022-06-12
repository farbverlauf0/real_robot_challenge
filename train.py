from trifinger_simulation import SimFinger, collision_objects
import pybullet
import numpy as np
from trifinger_simulation.tasks import move_cube_on_trajectory as task
from models import DDPG
from transformations import get_vector_state, vector_to_action

FINGER_TYPE = 'fingerpro'
EPS = 0.05
REWARD_SCALE = 1000


if __name__ == '__main__':
    sim = SimFinger(
        finger_type=FINGER_TYPE,
        enable_visualization=False,
        robot_position_offset=(0, 0, 0.02)
    )
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=0,
        cameraPitch=-30,
        cameraTargetPosition=(0, 0, 0.2),
        physicsClientId=sim._pybullet_client_id
    )
    np.random.seed(0)
    sim.reset_finger_positions_and_velocities(0.2 * np.random.randn(3))
    cube = collision_objects.Cube(position=[*np.random.uniform(-0.15, 0.15, 2), 0.0325])

    task.seed(0)
    task.EPISODE_LENGTH = 10000000
    goals = task.sample_goal()

    observation_dim = sim.number_of_fingers * 10 + len(cube.get_state()[0]) + len(task.get_active_goal(goals, 0))
    action_dim = sim.number_of_fingers * 6
    net = DDPG(observation_dim, action_dim)

    active_goal = task.get_active_goal(goals, 0)
    vec_state = get_vector_state(sim.get_observation(0), cube.get_state(), active_goal)
    current_distance = task.evaluate_state(goals, 0, cube.get_state()[0])

    while True:
        vec_desired_action = np.clip(net.act(vec_state) * 0.3 + np.random.randn(action_dim) * EPS, -0.396, 0.396)
        t = sim.append_desired_action(vector_to_action(vec_desired_action, sim))
        goal_is_changed = np.all(task.get_active_goal(goals, t + 1) != active_goal)
        if goal_is_changed:
            print('Goal has been changed!')
            active_goal = task.get_active_goal(goals, t + 1)
            cube.set_state(position=[*np.random.uniform(-0.15, 0.15, 2), 0.0325], orientation=(0, 0, 0, 1))
            sim.reset_finger_positions_and_velocities(0.2 * np.random.randn(3))
        vec_next_state = get_vector_state(sim.get_observation(t+1), cube.get_state(), active_goal)
        next_distance = task.evaluate_state(goals, t + 1, cube.get_state()[0])
        if not goal_is_changed:
            reward = (current_distance - next_distance) * REWARD_SCALE
            net.update((vec_state, vec_desired_action, vec_next_state, reward))
        vec_state = vec_next_state
        current_distance = next_distance
        if t % 1000 == 0:
            print(f'Step: {t}, Distance: {current_distance}')
            net.save()
        if t == task.EPISODE_LENGTH - 1:
            break

