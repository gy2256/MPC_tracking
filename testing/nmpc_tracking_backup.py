#!/usr/bin/env python
# -*- coding: utf-8 -*-

import casadi as ca

import numpy as np
import time
from draw import Draw_MPC_point_stabilization_v1


def step(dt, t0, x0, u, f):
    k1 = f(x0, u[:, 0])
    k2 = f(x0 + dt / 2 * k1, u[:, 0])
    k3 = f(x0 + dt / 2 * k2, u[:, 0])
    k4 = f(x0 + dt * k3, u[:, 0])
    state_next_ = x0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    t_ = t0 + dt
    u_next_ = ca.horzcat(u[1:, :], u[-1, :])

    return t_, state_next_, u_next_

if __name__ == "__main__":
    T = 0.1  # sampling time [s]
    N = 8  # prediction horizon
    rob_diam = 0.25 # [m]
    v_max = 0.6
    omega_max = np.pi / 4.0

    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    theta = ca.SX.sym("theta")
    states = ca.vertcat(x, y)
    states = ca.vertcat(states, theta)
    n_states = states.size()[0]

    v = ca.SX.sym("v")
    omega = ca.SX.sym("omega")
    controls = ca.vertcat(v, omega)
    n_controls = controls.size()[0]

    ## rhs
    rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta))
    rhs = ca.vertcat(rhs, omega)

    ## function
    f = ca.Function(
        "f", [states, controls], [rhs], ["input_state", "control_input"], ["rhs"]
    )

    ## for MPC
    U = ca.SX.sym("U", n_controls, N)
    X = ca.SX.sym("X", n_states, N + 1)
    P = ca.SX.sym("P", n_states + n_states)

    ### define
    X[:, 0] = P[:3]  # initial condiction

    #### define the relationship within the horizon
    for i in range(N):
        k1 = f(X[:, i], U[:, i])
        k2 = f(X[:, i] + T / 2 * k1, U[:, i])
        k3 = f(X[:, i] + T / 2 * k2, U[:, i])
        k4 = f(X[:, i] + T * k3, U[:, i])
        X[:, i + 1] = X[:, i] + T / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    ff = ca.Function("ff", [U, P], [X], ["input_U", "target_state"], ["horizon_states"])

    Q = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 1.5]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    #### cost function
    obj = 0
    for i in range(N):
        obj = (
            obj + (X[:, i] - P[3:]).T @ Q @ (X[:, i] - P[3:]) + U[:, i].T @ R @ U[:, i]
        )

    #### constrains
    g = []  # equal constrains
    for i in range(N + 1):
        g.append(X[0, i])
        g.append(X[1, i])
        

    nlp_prob = {
        "f": obj,
        "x": ca.reshape(U, -1, 1),
        "p": P,
        "g": ca.vcat(g),
    }  # here also can use ca.vcat(g) or ca.vertcat(*g)
    opts_setting = {
        "ipopt.max_iter": 100,
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.acceptable_tol": 1e-8,
        "ipopt.acceptable_obj_change_tol": 1e-6,
    }

    solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts_setting)

    # Simulation
    lbg = -2.0
    ubg = 2.0
    lbx = []
    ubx = []
    for _ in range(N):
        lbx.append(-v_max)
        ubx.append(v_max)
        lbx.append(-omega_max)
        ubx.append(omega_max)
    t0 = 0.0
    x0 = np.array([0.0, 0.0, 0.0]).reshape(-1, 1)  # initial state
    xs = np.array([1.5, 1.5, -np.pi/4]).reshape(-1, 1)  # final state
    u0 = np.array([0.0, 0.0] * N).reshape(-1, 2)  # np.ones((N, 2)) # controls
    x_c = []  # contains for the history of the state
    u_c = []
    t_c = []  # for the time
    xx = []
    sim_time = 20.0

    ## start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    c_p = np.concatenate((x0, xs))
    init_control = ca.reshape(u0, -1, 1)
    res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
    lam_x_ = res["lam_x"]
    ### inital test
    while np.linalg.norm(x0 - xs) > 1e-2 and mpciter - sim_time / T < 0.0:
        ## set parameter
        c_p = np.concatenate((x0, xs))
        init_control = ca.reshape(u0, -1, 1)
        t_ = time.time()
        res = solver(
            x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx, lam_x0=lam_x_
        )
        lam_x_ = res["lam_x"]
        # res = solver(x0=init_control, p=c_p,)
        # print(res['g'])
        index_t.append(time.time() - t_)
        u_sol = ca.reshape(
            res["x"], n_controls, N
        )  # one can only have this shape of the output
        ff_value = ff(u_sol, c_p)  # [n_states, N+1]
        x_c.append(ff_value)
        u_c.append(u_sol[:, 0])
        t_c.append(t0)
        t0, x0, u0 = step(T, t0, x0, u_sol, f)

        x0 = ca.reshape(x0, -1, 1)
        xx.append(x0.full())
        mpciter = mpciter + 1
    t_v = np.array(index_t)
    print(t_v.mean())
    print((time.time() - start_time) / (mpciter))
    draw_result = Draw_MPC_point_stabilization_v1(
        rob_diam=0.3,
        init_state=x0.full(),
        target_state=xs,
        robot_states=xx,
        export_fig=False,
    )
