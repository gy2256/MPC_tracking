
import numpy as np
from nmpc_tracking_opti_framework import NMPC_Unicycle
from Utilits.utils import Animate_unicycle_robot, step_unicycle, waypoints_to_x_ref, calculate_unicycle_local_reference


if __name__ == "__main__":
    state_weight = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.1]])
    control_weight = np.array([[0.5, 0.0], [0.0, 0.05]])
    initial_state = np.array([0.0, 0.0, 0.0])
    current_state = initial_state
    obstacles = []
    #target_state = np.array([1.5, 1.5, -np.pi / 4.0])
    target_speed = 1.5
    Receding_horizon_N = 6
    robot_radius = 0.25
    dt = 0.1
    current_time = 0.0 #time, float
    t_step = 0  # time step, integer
    target_index = 0 # target index for tracking_mode = "closest_interpolated_point"
    simulation_max_time = 50.0
    Unicycle = NMPC_Unicycle(Receding_horizon_N, dt, state_weight, control_weight, initial_state,obstacles,robot_radius)

    waypoints = np.array([[initial_state[0], initial_state[1]], [1.5, 1.5], [2.5, 1.5], [2.7, 2.5]])
    x_ref = waypoints_to_x_ref(waypoints, target_speed, interpolation_type="linear", model="Unicycle") # x_ref = [[x, y, theta],...,] for unicycle model

    state_history = [initial_state]
    control_history = []
    time_history = []

    
    while current_time < simulation_max_time:
        x_local_ref, target_index = calculate_unicycle_local_reference(x_ref, t_step, current_state, Receding_horizon_N,
                                                              target_index=target_index)
        
        target_state = x_local_ref# To Do : change mpc_control to update with sequence of x_local_ref, instead of a single target_state

        u_sol = Unicycle.mpc_control(current_state, target_state)
        next_t, next_state, = step_unicycle(dt, current_time, current_state, u_sol)

        state_history.append(next_state)
        control_history.append(u_sol)
        time_history.append(next_t)

        current_state = next_state
        current_time += dt
        t_step += 1

    state_boundary = [-2, 5, -2, 5]  # [x_min, x_max, y_min, y_max]
    animator = Animate_unicycle_robot(x_ref, initial_state, target_state[:,0], state_history, robot_radius, obstacles, state_boundary)
    animator.save_animation("MPC_unicycle_robot.mp4")








