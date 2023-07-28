import matplotlib.pyplot as plt
import numpy as np
import scipy
import matplotlib.animation as animation
import matplotlib.patches as mpatches

def read_waypoints(path_to_waypoints_file):
    waypoints = np.load(path_to_waypoints_file, allow_pickle=True)
    return waypoints

def nearest_interpolated_point(x_ref, x_current, target_index):
        # find nearest point on the interpolated trajectory
    N_indx_search = 10  # search for the next 10 point
    dx = [x_current[0] - icx for icx in x_ref[0][target_index:target_index + N_indx_search]]
    dy = [x_current[2] - icy for icy in x_ref[2][target_index:target_index + N_indx_search]]
    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
    mind = min(d)
    ind = d.index(mind) + target_index

    return ind

def calculate_local_reference(x_ref, t_step, x_current, N ,target_index=0,
                                  tracking_mode="closest_interpolated_point"):
    state_dimension = len(x_ref[0])

    if tracking_mode == "running_time":
        # Create an x_local_ref as a shifting window over time.
        x_local_ref = np.zeros((state_dimension, N + 1))

        if t_step < len(x_ref[0]) - N - 1:
            for i in range(N + 1):
                x_local_ref[:, i] = x_ref[:, i + t_step]
        else:
            for i in range(N + 1):
                x_local_ref[:, i] = x_ref[:, -1]
    elif tracking_mode == "closest_interpolated_point":
        x_local_ref = np.zeros((state_dimension , N + 1))
        total_interpolated_points_len = len(x_ref[0])

        ind = nearest_interpolated_point(x_ref, x_current, target_index)
        if target_index >= ind:
            ind = target_index

        x_local_ref[0, 0] = x_ref[0, ind]
        x_local_ref[1, 0] = x_ref[1, ind]
        x_local_ref[2, 0] = x_ref[2, ind]
        x_local_ref[3, 0] = x_ref[3, ind]

        for i in range(N + 1):
            if (ind + N) < total_interpolated_points_len:
                x_local_ref[0, i] = x_ref[0, ind + i]
                x_local_ref[1, i] = x_ref[1, ind + i]
                x_local_ref[2, i] = x_ref[2, ind + i]
                x_local_ref[3, i] = x_ref[3, ind + i]
            else:
                x_local_ref[0, i] = x_ref[0, -1]
                x_local_ref[1, i] = x_ref[1, -1]
                x_local_ref[2, i] = x_ref[2, -1]
                x_local_ref[3, i] = x_ref[3, -1]

    return x_local_ref, ind  # this ind is the target index for the next time step

def step_unicycle(dt, current_t, current_state, u):
    #u = u[:,0]
    def ode_func(t, x):
        vel, yaw = u[0], u[1]
        theta = x[2]
        dx = np.array([np.cos(theta) * vel, np.sin(theta) * vel, yaw])
        return dx

    sol = scipy.integrate.solve_ivp(ode_func, [0, dt], current_state, method="RK45")
    next_state = sol.y[:, -1]
    next_t = current_t + dt

    return next_t, next_state

class Animate_robot():
    def __init__(self, init_state, target_state, state_history,robot_radius):
        self.init_state = init_state
        self.target_state = target_state
        self.robot_radius = robot_radius
        self.state_history = state_history
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-3.0, 3.0), ylim=(-3.0, 3.0))
        self.animation_init()
        self.animation = animation.FuncAnimation(
            self.fig,
            self.animation_loop,
            range(len(self.state_history)),
            init_func=self.animation_init,
            interval=100,
            repeat=False,
        )

        plt.show()


    def animation_init(self):
        self.target_circle = plt.Circle(
            self.target_state[:2], self.robot_radius, color="b", fill=False
        )
        self.ax.add_artist(self.target_circle)

        self.target_arr = mpatches.Arrow(
            self.target_state[0],
            self.target_state[1],
            self.robot_radius * np.cos(self.target_state[2]),
            self.robot_radius * np.sin(self.target_state[2]),
            width=0.2,
            color="b",
        )
        self.ax.add_patch(self.target_arr)

        self.robot_body = plt.Circle(
            self.init_state[:2], self.robot_radius, color="r", fill=False
        )
        self.ax.add_artist(self.robot_body)

        self.robot_arr = mpatches.Arrow(
            self.init_state[0],
            self.init_state[1],
            self.robot_radius * np.cos(self.init_state[2]),
            self.robot_radius * np.sin(self.init_state[2]),
            width=0.2,
            color="r",
        )
        self.ax.add_patch(self.robot_arr)


    def animation_loop(self, indx):
        position = self.state_history[indx][:2]
        orientation = self.state_history[indx][2]
        self.robot_body.remove()
        self.robot_body = plt.Circle(
            orientation, self.robot_radius, color="r", fill=False
        )
        self.ax.add_artist(self.robot_body)
        self.robot_body.center = position
        self.robot_arr.remove()
        self.robot_arr = mpatches.Arrow(
            position[0],
            position[1],
            self.robot_radius * np.cos(orientation),
            self.robot_radius * np.sin(orientation),
            width=0.2,
            color="r",
        )
        self.ax.add_patch(self.robot_arr)
        return self.robot_arr, self.robot_body


