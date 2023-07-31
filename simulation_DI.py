
import numpy as np
from nmpc_tracking import NMPC_Double_Integrator
from Utilits.utils import Animate_double_integrator_robot, step_double_integrator, waypoints_to_x_ref, calculate_local_reference


if __name__ == "__main__":
    state_weight = np.diag([6.0, 1.0, 6.0, 1.0])
    control_weight = np.diag([1.0, 1.0])
    initial_state = np.array([0.0, 0.0, 0.0, 0.0])
    current_state = initial_state
    
    target_speed = 1.0
    Receding_horizon_N = 6
    dt = 0.1
    current_time = 0.0 #time, float
    t_step = 0  # time step, integer
    target_index = 0 # target index for tracking_mode = "closest_interpolated_point"
    simulation_max_time = 50.0
    Double_integrator = NMPC_Double_Integrator(Receding_horizon_N, dt, state_weight, control_weight, initial_state)

    waypoints = np.array([[initial_state[0], initial_state[2]], [1.5, 1.5], [2.5, 1.5], [-2.7, 2.5]])
    x_ref = waypoints_to_x_ref(waypoints, target_speed, interpolation_type="linear", model="Double_Integrators") # x_ref = [[x, y, theta],...,] for unicycle model

    state_history = [initial_state]
    control_history = []
    time_history = []
    robot_radius = 0.25
    
    while current_time < simulation_max_time:
        x_local_ref, target_index = calculate_local_reference(x_ref, t_step, current_state, Receding_horizon_N,
                                                              target_index=target_index)
        
        target_state = x_local_ref# To Do : change mpc_control to update with sequence of x_local_ref, instead of a single target_state

        u_sol = Double_integrator.mpc_control(current_state, target_state)
        next_t, next_state, = step_double_integrator(dt, current_time, current_state, u_sol)

        state_history.append(next_state)
        control_history.append(u_sol)
        time_history.append(next_t)

        current_state = next_state
        current_time += dt
        t_step += 1


    Animate_double_integrator_robot(initial_state, state_history, robot_radius)








