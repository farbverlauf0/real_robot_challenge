import torch
from trifinger_simulation import SimFinger, collision_objects
import pybullet
from trifinger_simulation.tasks import move_cube_on_trajectory as task
from transformations import *


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl", map_location=torch.device('cpu'))

    def act(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float)
        return self.model(state).detach().numpy()[0].reshape(-1)


if __name__ == '__main__':
    sim = SimFinger(
        finger_type='fingerpro',
        enable_visualization=True,
        robot_position_offset=(0, 0, 0.02)
    )
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=2.0,
        cameraYaw=0.0,
        cameraPitch=-30.0,
        cameraTargetPosition=(0, 0, 0.2),
        physicsClientId=sim._pybullet_client_id,
    )
    np.random.seed(0)
    cube = collision_objects.Cube(position=[*np.random.uniform(-0.15, 0.15, 2), 0.0325])
    task.seed(0)
    task.EPISODE_LENGTH = 120000
    goals = task.sample_goal()

    agent = Agent()
    vec_state = get_vector_state(sim.get_observation(0), cube.get_state(), task.get_active_goal(goals, 0))

    while True:
        vec_desired_action = agent.act(vec_state) * 0.3
        a = vector_to_action(vec_desired_action, sim)
        t = sim.append_desired_action(a)
        vec_state = get_vector_state(sim.get_observation(t + 1), cube.get_state(), task.get_active_goal(goals, t + 1))
        if t % 10000 == 0:
            print(f'Step: {t}, Distance: {task.evaluate_state(goals, t + 1, cube.get_state()[0])}')
        if t == task.EPISODE_LENGTH - 1:
            break

