#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca

import numpy as np

"""
Created on July 25, 2023
by Guang Yang

MPC implementation for unicycle model
"""

class NMPC_Double_Integrator:
    def __init__(self, N, dt, state_weight, control_weight, init_state) -> None:
        self.dt = dt
        self.N = N
        self.Q = state_weight
        self.R = control_weight
        self.init_state = init_state

        # State related variable initialization
        self.x1 = ca.SX.sym("x1")
        self.v1 = ca.SX.sym("v1")
        self.x2 = ca.SX.sym("x2")
        self.v2 = ca.SX.sym("v2")
        self.states = ca.vertcat(self.x1, self.v1, self.x2, self.v2)
        self.n_states = self.states.size()[0]

        # Control related variable initialization
        self.u1 = ca.SX.sym("u1")
        self.u2 = ca.SX.sym("u2")
        self.controls = ca.vertcat(self.u1, self.u2)
        self.n_controls = self.controls.size()[0]
        self.u_max = 20  # control u1

        self.dynammics_f = ca.vertcat(
            self.v1, self.u1, self.v2, self.u2
        )

        # MPC Solver Initialization
        self.f = ca.Function(
            "f",
            [self.states, self.controls],
            [self.dynammics_f],
            ["input_state", "control_input"],
            ["dynamics"],
        )
        self.U = ca.SX.sym("U", self.n_controls, self.N)
        self.X = ca.SX.sym("X", self.n_states, self.N + 1)
        
        #self.P = ca.SX.sym("P", self.n_states + self.n_states)
        #self.P = ca.SX.sym("P", self.n_states + self.n_states*self.N) # P[0:3] = x0, P[3:] = x_ref = [x1[t_0],x2[t_0],x3[t_0],x1[t1],x2[t2],...]
        self.P = ca.SX.sym("P", self.n_states)

        self.goal_dist = 0.2  # Termination Condition
        self.max_simulation_time = 20.0  # Termination Condition

    def mpc_control(self, x_current, x_ref):
        """
        Input: x_ref: reference trajectory

        Output: State Trajectory and Control Trajectory for the next N steps
        """
        # initial condiction

        x0 = x_current.reshape(-1, 1)  # x_current = np.array([0.0, 0.0, 0.0])
        #xs = x_ref.reshape(-1, 1) #x_ref is a sequence of states, this needs to be modified
        #c_p = np.concatenate((x0, xs))
        c_p = x0
        u0 = np.array([0.0, 0.0] * self.N).reshape(-1, self.n_controls) # np.ones((N, 2)) # controls

        self.X[:, 0] = self.P[:4]

        for i in range(self.N):
            k1 = self.f(self.X[:, i], self.U[:, i])
            k2 = self.f(self.X[:, i] + self.dt / 2 * k1, self.U[:, i])
            k3 = self.f(self.X[:, i] + self.dt / 2 * k2, self.U[:, i])
            k4 = self.f(self.X[:, i] + self.dt * k3, self.U[:, i])
            self.X[:, i + 1] = self.X[:, i] + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # cost function
        cost = 0
        '''
        for i in range(self.N):
            cost = (
                cost
                + (self.X[:, i] - self.P[3:]).T @ self.Q @ (self.X[:, i] - self.P[3:])
                + self.U[:, i].T @ self.R @ self.U[:, i]
            )
        '''
        for i in range(1, self.N):
            cost = (
                cost + (self.X[:,i]-x_ref[:,i]).T @ self.Q @ (self.X[:,i]-x_ref[:,i]) 
                + self.U[:,i].T @ self.R @ self.U[:,i]
            )

        # Constraints

        g = []
        for i in range(self.N + 1):
            g.append(self.X[0, i])
            g.append(self.X[2, i])

        lbx = [] # lower bound of control
        ubx = [] # upper bound of control
        for _ in range(self.N*self.n_controls):
            lbx.append(-self.u_max)
            ubx.append(self.u_max)

        lbg = -20.0
        ubg = 20.0

        nlp_prob = {
            "f": cost,
            "x": ca.reshape(self.U, -1, 1), # decision vairble u
            "p": self.P, # parameter P = [x0, xs] current state, reference state
            "g": ca.vcat(g), # dynamics constraint
        }

        opts_setting = {
            "ipopt.max_iter": 100,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-8,
            "ipopt.acceptable_obj_change_tol": 1e-6,
        }

        solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts_setting)
        init_control = ca.reshape(u0, -1, 1)

        result = solver(
            x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx
        )

        u_sol = ca.reshape(result["x"], self.n_controls, self.N)
        u_sol = u_sol[:, 0].toarray()
        u_sol = u_sol[:, 0]

        return u_sol  # Return the first control


class NMPC_Unicycle:
    def __init__(self, N, dt, state_weight, control_weight, init_state) -> None:
        self.dt = dt
        self.N = N
        self.Q = state_weight
        self.R = control_weight
        self.init_state = init_state

        # State related variable initialization
        self.x = ca.SX.sym("x")
        self.y = ca.SX.sym("y")
        self.theta = ca.SX.sym("theta")
        self.states = ca.vertcat(self.x, self.y)
        self.states = ca.vertcat(self.states, self.theta)
        self.n_states = self.states.size()[0]

        # Control related variable initialization
        self.v = ca.SX.sym("v")
        self.omega = ca.SX.sym("omega")
        self.controls = ca.vertcat(self.v, self.omega)
        self.n_controls = self.controls.size()[0]
        self.v_max = 1.5  # control u1
        self.omega_max = np.pi / 4.0  # control u2

        self.dynammics_f = ca.vertcat(
            self.v * ca.cos(self.theta), self.v * ca.sin(self.theta)
        )
        self.dynammics_f = ca.vertcat(self.dynammics_f, self.omega)

        # MPC Solver Initialization
        self.f = ca.Function(
            "f",
            [self.states, self.controls],
            [self.dynammics_f],
            ["input_state", "control_input"],
            ["dynamics"],
        )
        self.U = ca.SX.sym("U", self.n_controls, self.N)
        self.X = ca.SX.sym("X", self.n_states, self.N + 1)
        
        #self.P = ca.SX.sym("P", self.n_states + self.n_states)
        #self.P = ca.SX.sym("P", self.n_states + self.n_states*self.N) # P[0:3] = x0, P[3:] = x_ref = [x1[t_0],x2[t_0],x3[t_0],x1[t1],x2[t2],...]
        self.P = ca.SX.sym("P", self.n_states)

        self.goal_dist = 0.2  # Termination Condition
        self.max_simulation_time = 20.0  # Termination Condition

    def mpc_control(self, x_current, x_ref):
        """
        Input: x_ref: reference trajectory

        Output: State Trajectory and Control Trajectory for the next N steps
        """
        # initial condiction

        x0 = x_current.reshape(-1, 1)  # x_current = np.array([0.0, 0.0, 0.0])
        #xs = x_ref.reshape(-1, 1) #x_ref is a sequence of states, this needs to be modified
        #c_p = np.concatenate((x0, xs))
        c_p = x0
        u0 = np.array([0.0, 0.0] * self.N).reshape(-1, self.n_controls) # np.ones((N, 2)) # controls

        self.X[:, 0] = self.P[:3]

        for i in range(self.N):
            k1 = self.f(self.X[:, i], self.U[:, i])
            k2 = self.f(self.X[:, i] + self.dt / 2 * k1, self.U[:, i])
            k3 = self.f(self.X[:, i] + self.dt / 2 * k2, self.U[:, i])
            k4 = self.f(self.X[:, i] + self.dt * k3, self.U[:, i])
            self.X[:, i + 1] = self.X[:, i] + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        ff = ca.Function(
            "ff",
            [self.U, self.P],
            [self.X],
            ["input_U", "target_state"],
            ["horizon_states"],
        )

        # cost function
        cost = 0
        '''
        for i in range(self.N):
            cost = (
                cost
                + (self.X[:, i] - self.P[3:]).T @ self.Q @ (self.X[:, i] - self.P[3:])
                + self.U[:, i].T @ self.R @ self.U[:, i]
            )
        '''
        for i in range(1, self.N):
            cost = (
                cost + (self.X[:,i]-x_ref[:,i]).T @ self.Q @ (self.X[:,i]-x_ref[:,i]) 
                + self.U[:,i].T @ self.R @ self.U[:,i]
            )

        # Constraints

        g = []
        for i in range(self.N + 1):
            g.append(self.X[0, i])
            g.append(self.X[1, i])

        lbx = [] # lower bound of control
        ubx = [] # upper bound of control
        for _ in range(self.N):
            lbx.append(-self.v_max)
            lbx.append(-self.omega_max)
            ubx.append(self.v_max)
            ubx.append(self.omega_max)

        lbg = -10.0
        ubg = 10.0

        nlp_prob = {
            "f": cost,
            "x": ca.reshape(self.U, -1, 1), # decision vairble u
            "p": self.P, # parameter P = [x0, xs] current state, reference state
            "g": ca.vcat(g), # dynamics constraint
        }

        opts_setting = {
            "ipopt.max_iter": 100,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.acceptable_tol": 1e-8,
            "ipopt.acceptable_obj_change_tol": 1e-6,
        }

        solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts_setting)
        init_control = ca.reshape(u0, -1, 1)

        result = solver(
            x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx
        )

        u_sol = ca.reshape(result["x"], self.n_controls, self.N)
        u_sol = u_sol[:, 0].toarray()
        u_sol = u_sol[:,0]

        return u_sol  # Return the first control
    


if __name__ == "__main__":
    state_weight = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.1]])
    control_weight = np.array([[0.5, 0.0], [0.0, 0.05]])
    initial_state = np.array([0.0, 0.0, 0.0])
    target_state = np.array([1.5, 1.5, -np.pi/4.0])
    Receding_horizon_N = 8
    dt = 0.1
    
    Unicycle = NMPC_Unicycle(Receding_horizon_N, dt, state_weight, control_weight, initial_state)
    u_sol = Unicycle.mpc_control(initial_state, target_state)
