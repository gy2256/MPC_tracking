#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca
from casadi import Opti
import numpy as np

"""
Created on July 25, 2023
by Guang Yang

MPC implementation for unicycle model
"""

class NMPC_Double_Integrator:
    def __init__(self, N, dt, state_weight, control_weight, init_state, obstacles, robot_radius) -> None:
        self.dt = dt
        self.N = N
        self.Q = state_weight
        self.R = control_weight
        self.init_state = init_state
        self.robot_radius = robot_radius

        self.obstacles = obstacles # for dCBF constraints
        self.k_cbf = 1.0 #CBF parameter
        self.n_states = 4
        self.n_controls = 2

        # State related variable initialization
        self.f = lambda xk, uk: ca.vertcat(xk[1], uk[0], xk[3], uk[1])
        self.u_max = 20
        self.goal_dist = 0.2  # Termination Condition
        self.max_simulation_time = 20.0  # Termination Condition


    def mpc_control(self, x_current, x_ref):
        self.opti = Opti()  # Reinitialize the problem to clear previous solution
        self.X = self.opti.variable(self.n_states, self.N + 1)  # state trajectory
        self.U = self.opti.variable(self.n_controls, self.N)  # control trajectory
        # Initial State Constraint
        x_current = x_current.reshape(-1, 1)
        self.opti.subject_to(self.X[:, 0] == x_current) # initial state constraint

        #Control bounds Constraint
        self.opti.subject_to(self.opti.bounded(-self.u_max, self.U[0, :], self.u_max)) # control input constraint
        self.opti.subject_to(self.opti.bounded(-self.u_max, self.U[1, :], self.u_max))

        #Dynamics Constraint using RK4, TO DO: switch to closed form to make the problem feasible?
        for k in range(self.N):
            k1 = self.f(self.X[:, k], self.U[:, k])
            k2 = self.f(self.X[:, k] + self.dt / 2 * k1, self.U[:, k])
            k3 = self.f(self.X[:, k] + self.dt / 2 * k2, self.U[:, k])
            k4 = self.f(self.X[:, k] + self.dt * k3, self.U[:, k])
            x_next = self.X[:, k] + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.opti.subject_to(self.X[:, k + 1] == x_next)

        # CBF constraint for circular obstacle
        for obs in self.obstacles:
            for i in range(self.N):
                h = (self.X[0, i] - obs[0])**2 + (self.X[2, i] - obs[1])**2 - (obs[2]+self.robot_radius)**2
                h_next = (self.X[0, i+1] - obs[0])**2 + (self.X[2, i+1] - obs[1])**2 - (obs[2]+self.robot_radius)**2

                self.opti.subject_to(h_next - h + self.k_cbf * h >= 0)

        # cost function
        cost = 0
        for i in range(self.N):
            cost += (self.X[:, i]-x_ref[:,i]).T @ self.Q @ (self.X[:, i]-x_ref[:,i]) + self.U[:, i].T @ self.R @ self.U[:, i]
        self.opti.minimize(cost)

        solver_option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}

        self.opti.solver("ipopt",solver_option)
        self.solution = self.opti.solve()
        u_sol = self.solution.value(self.U)
        u_sol = u_sol[:, 0]

        del self.opti

        return u_sol  # Return the first control


class NMPC_Unicycle:
    def __init__(self, N, dt, state_weight, control_weight, init_state, obstacles, robot_radius) -> None:
        self.dt = dt
        self.N = N
        self.Q = state_weight
        self.R = control_weight
        self.init_state = init_state
        self.obstacles = obstacles # for dCBF constraints
        self.robot_radius = robot_radius

        self.k_cbf = 1.0  # CBF parameter
        self.n_states = 3
        self.n_controls = 2

        # State related variable initialization
        self.f = lambda xk, uk: ca.vertcat(uk[0]*np.cos(xk[2]), uk[0]*np.sin(xk[2]), uk[1])
        self.v_max = 1.5
        self.omega_max = np.pi/4
        self.goal_dist = 0.2  # Termination Condition
        self.max_simulation_time = 20.0  # Termination Condition



    def mpc_control(self, x_current, x_ref):
        self.opti = Opti()  # Reinitialize the problem to clear previous solution
        self.X = self.opti.variable(self.n_states, self.N + 1)  # state trajectory
        self.U = self.opti.variable(self.n_controls, self.N)  # control trajectory
        # Initial State Constraint
        x_current = x_current.reshape(-1, 1)
        self.opti.subject_to(self.X[:, 0] == x_current)  # initial state constraint

        # Control bounds Constraint
        self.opti.subject_to(self.opti.bounded(-self.v_max, self.U[0, :], self.v_max))  # control input constraint
        self.opti.subject_to(self.opti.bounded(-self.omega_max, self.U[1, :], self.omega_max))

        # Dynamics Constraint using RK4, TO DO: switch to closed form to make the problem feasible?
        for k in range(self.N):
            k1 = self.f(self.X[:, k], self.U[:, k])
            k2 = self.f(self.X[:, k] + self.dt / 2 * k1, self.U[:, k])
            k3 = self.f(self.X[:, k] + self.dt / 2 * k2, self.U[:, k])
            k4 = self.f(self.X[:, k] + self.dt * k3, self.U[:, k])
            x_next = self.X[:, k] + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            self.opti.subject_to(self.X[:, k + 1] == x_next)

        # CBF constraint for circular obstacle
        for obs in self.obstacles:
            for i in range(self.N):
                h = (self.X[0, i] - obs[0]) ** 2 + (self.X[2, i] - obs[1]) ** 2 - (obs[2] + self.robot_radius) ** 2
                h_next = (self.X[0, i + 1] - obs[0]) ** 2 + (self.X[2, i + 1] - obs[1]) ** 2 - (
                            obs[2] + self.robot_radius) ** 2

                self.opti.subject_to(h_next - h + self.k_cbf * h >= 0)

        # cost function
        cost = 0
        for i in range(self.N):
            cost += (self.X[:, i] - x_ref[:, i]).T @ self.Q @ (self.X[:, i] - x_ref[:, i]) + self.U[:,
                                                                                             i].T @ self.R @ self.U[:,
                                                                                                             i]
        self.opti.minimize(cost)

        solver_option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}

        self.opti.solver("ipopt", solver_option)
        self.solution = self.opti.solve()
        u_sol = self.solution.value(self.U)
        u_sol = u_sol[:, 0]

        del self.opti
        return u_sol  # Return the first control
    


if __name__ == "__main__":
    state_weight = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.1]])
    control_weight = np.array([[0.5, 0.0], [0.0, 0.05]])
    initial_state = np.array([0.0, 0.0, 0.0])
    target_state = np.array([[1.5, 1.5, -np.pi/4.0],[1.5, 1.5, -np.pi/4.0],[1.5, 1.5, -np.pi/4.0]])
    Receding_horizon_N = 3
    obstacles = []
    dt = 0.1
    robot_radius = 0.25
    
    Unicycle = NMPC_Unicycle(Receding_horizon_N, dt, state_weight, control_weight, initial_state, obstacles,
                             robot_radius)
    u_sol = Unicycle.mpc_control(initial_state, target_state)
    print(u_sol)
