import matplotlib.pyplot as plt
import numpy as np
import scipy
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from Utilits.CubicSpline import cubic_spline_planner, spline_continuity

def read_waypoints(path_to_waypoints_file):
    waypoints = np.load(path_to_waypoints_file, allow_pickle=True)
    return waypoints

def calculate_velocity_between_interpolated_positions(interpolated_x, interpolated_y, target_speed):
        v_interpolated_x = np.zeros(len(interpolated_x))
        v_interpolated_y = np.zeros(len(interpolated_y))

        for i in range(len(interpolated_x)-1):
            delta_interpolated_x = interpolated_x[i+1] - interpolated_x[i]
            delta_interpolated_y = interpolated_y[i+1] - interpolated_y[i]
            angle = np.arctan2(delta_interpolated_y, delta_interpolated_x)
            v_interpolated_x[i] = target_speed*np.cos(angle)
            v_interpolated_y[i] = target_speed*np.sin(angle)

        return v_interpolated_x, v_interpolated_y

def waypoints_to_x_ref(waypoints, target_speed, interpolation_type="linear", model="Double_Integrators"):
        interpolated_dist = 0.1
        if interpolation_type == "linear":
            rx, ry, ryaw = [], [], []
            sp = spline_continuity.Spline2D(x=waypoints[:, [0]].flatten(), y=waypoints[:, [1]].flatten(),
                                            kind="linear")
            s = np.arange(0, sp.s[-1], interpolated_dist)
            
            for idx, i_s in enumerate(s):
                ix, iy = sp.calc_position(i_s)

                if idx < len(s)-1:
                    next_ix, next_iy = sp.calc_position(s[idx+1])
                    yaw = np.arctan((next_iy-iy)/(next_ix-ix))
                else:
                    yaw = 0.0
                
                rx.append(ix)
                ry.append(iy)
                ryaw.append(yaw) # To DO : calculate correct yaw
                

            if model == "Double_Integrators":
                state_dimension = 4
                interpolated_x1 = np.array(rx)
                interpolated_x3 = np.array(ry)
            elif model == "Unicycle":
                state_dimension = 3
                interpolated_x1 = np.array(rx)
                interpolated_x2 = np.array(ry)
                interpolated_x3 = np.array(ryaw)

        elif interpolation_type == "cubic":
            if model == "Double_Integrators":
                interpolated_x1, interpolated_x3, _, _, _ = cubic_spline_planner.calc_spline_course(
                    x=waypoints[:, [0]].flatten(),
                    y=waypoints[:, [1]].flatten(),
                    ds=interpolated_dist)  # ds is the distance between interpolated points
            
            elif model == "Unicycle":
                interpolated_x1, interpolated_x2, interpolated_x3, _, _ = cubic_spline_planner.calc_spline_course(
                    x=waypoints[:, [0]].flatten(),
                    y=waypoints[:, [1]].flatten(),
                    ds=interpolated_dist)

        if model == "Double_Integrators":
            state_dimension = 4
            x_ref = np.zeros((state_dimension, len(interpolated_x1)))
            # Fill reference trajectory with interpolated positions in x1 and x3
            x_ref[0, :] = interpolated_x1[:]
            x_ref[2, :] = interpolated_x3[:]

            # Calculate Velocity between interpolated positions and fill into the x_ref
            v_x2, v_x4 = calculate_velocity_between_interpolated_positions(interpolated_x1, interpolated_x3,
                                                                            target_speed)
            x_ref[1, :] = v_x2[:]
            x_ref[3, :] = v_x4[:]

            # Set the speed at the goal to be zero
            x_ref[1, -1] = 0.0
            x_ref[3, -1] = 0.0

        elif model == "Unicycle":
            state_dimension = 3
            x_ref = np.zeros((state_dimension, len(interpolated_x1)))
            # Fill reference trajectory with interpolated positions in x1 and x3
            x_ref[0, :] = interpolated_x1[:]
            x_ref[1, :] = interpolated_x2[:]
            x_ref[2, :] = interpolated_x3[:]

        return x_ref

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
    state_dimension = len(x_ref[:,0])
    x_local_ref = np.zeros((state_dimension, N + 1))

    if tracking_mode == "running_time":
        # Create an x_local_ref as a shifting window over time.

        if t_step < len(x_ref[0]) - N - 1:
            for i in range(N + 1):
                x_local_ref[:, i] = x_ref[:, i + t_step]
        else:
            for i in range(N + 1):
                x_local_ref[:, i] = x_ref[:, -1]
    elif tracking_mode == "closest_interpolated_point":
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

def calculate_unicycle_local_reference(x_ref, t_step, x_current, N ,target_index=0):
    state_dimension = len(x_ref[:,0])

    x_local_ref = np.zeros((state_dimension , N + 1))
    total_interpolated_points_len = len(x_ref[0])

    ind = nearest_interpolated_point(x_ref, x_current, target_index)
    if target_index >= ind:
        ind = target_index

    x_local_ref[0, 0] = x_ref[0, ind]
    x_local_ref[1, 0] = x_ref[1, ind]
    x_local_ref[2, 0] = x_ref[2, ind]


    for i in range(N + 1):
        if (ind + N) < total_interpolated_points_len:
            x_local_ref[0, i] = x_ref[0, ind + i]
            x_local_ref[1, i] = x_ref[1, ind + i]
            x_local_ref[2, i] = x_ref[2, ind + i]

        else:
            x_local_ref[0, i] = x_ref[0, -1]
            x_local_ref[1, i] = x_ref[1, -1]
            x_local_ref[2, i] = x_ref[2, -1]


    return x_local_ref, ind  # this ind is the target index for the next time step

def step_unicycle(dt, current_t, current_state, u):
    def ode_func(t, x):
        vel, yaw = u[0], u[1]
        theta = x[2]
        dx = np.array([np.cos(theta) * vel, np.sin(theta) * vel, yaw])
        return dx

    sol = scipy.integrate.solve_ivp(ode_func, [0, dt], current_state, method="RK45")
    next_state = sol.y[:, -1]
    next_t = current_t + dt

    return next_t, next_state

def step_double_integrator(dt, current_t, current_state, u):
    def ode_func(t, x):
        # Double Integrator dynamics
        dx = np.array([x[1], u[0], x[3], u[1]])
        return dx

    sol = scipy.integrate.solve_ivp(ode_func, [0, dt], current_state, method="RK45")
    next_state = sol.y[:, -1]
    next_t = current_t + dt

    return next_t, next_state

class Animate_unicycle_robot():
    def __init__(self, init_state, target_state, state_history, robot_radius):
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

class Animate_double_integrator_robot():
    def __init__(self, init_state, state_history, robot_radius):
        self.init_state = init_state
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

        self.robot_body = plt.Circle(
            self.init_state[:2], self.robot_radius, color="r", fill=False
        )
        self.ax.add_artist(self.robot_body)



    def animation_loop(self, indx):
        position_x1 = self.state_history[indx][0]
        position_x2 = self.state_history[indx][2]

        self.robot_body.remove()
        self.robot_body = plt.Circle(
            (position_x1,position_x2), self.robot_radius, color="r", fill=False
        )
        self.ax.add_artist(self.robot_body)
        self.robot_body.center = [position_x1, position_x2]

        return self.robot_body


