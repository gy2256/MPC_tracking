
import numpy as np
from nmpc_tracking import NMPC_Unicycle
from Utilits.utils import Animate_robot, step_unicycle


if __name__ == "__main__":
    state_weight = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.1]])
    control_weight = np.array([[0.5, 0.0], [0.0, 0.05]])
    initial_state = np.array([0.0, 0.0, 0.0])
    current_state = initial_state
    target_state = np.array([1.5, 1.5, -np.pi / 4.0])
    Receding_horizon_N = 8
    dt = 0.1
    current_time = 0.0
    simulation_max_time = 10.0
    Unicycle = NMPC_Unicycle(Receding_horizon_N, dt, state_weight, control_weight, initial_state)

    state_history = [initial_state]
    control_history = []
    time_history = []

    robot_radius = 0.25

    while current_time < simulation_max_time:
        u_sol = Unicycle.mpc_control(current_state, target_state)

        next_t, next_state, = step_unicycle(dt, current_time, current_state, u_sol)

        state_history.append(next_state)
        control_history.append(u_sol)
        time_history.append(next_t)

        current_state = next_state
        current_time += dt

    Animate_robot(initial_state, target_state, state_history, robot_radius)








