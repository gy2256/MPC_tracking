import numpy as np
import matplotlib.pyplot as plt

# System dynamics (plant)
def plant_dynamics(x, u):
    A = np.array([[-0.2, 0.4], [-0.6, -0.3]])
    B = np.array([[0.3], [0.2]])
    dx = np.dot(A, x) + np.dot(B, u)
    return dx[:, 0]

# Reference model
def reference_model(xr, u):
    Ar = np.array([[-0.4, 0.6], [-0.6, -0.4]])
    Br = np.array([[0.5], [0.5]])
    dxr = np.dot(Ar, xr) + np.dot(Br, u)
    return dxr[:, 0]

# Adaptive controller
class AdaptiveController:
    def __init__(self):
        self.theta = np.zeros((4, 1))  # Adaptive parameters
        self.k = 5.0                   # Controller gain

    def control_law(self, x, xr):
        # State and control input
        x1, x2 = x
        xr1, xr2 = xr

        # Parameter update law
        Phi = np.array([[x1, x2, -xr1 * x1, -xr1 * x2],
                        [x2, -x1, -xr2 * x1, -xr2 * x2]])
        dtheta = -self.k * np.dot(Phi.T, np.array([[x1], [x2]]))

        # Adaptive parameter update
        self.theta += dtheta

        # Control input
        u = np.dot(self.theta.T, np.array([[x1], [x2], [xr1], [xr2]]))

        return u

# Simulation parameters
dt = 0.01                  # Time step
t_final = 5.0              # Final time
t = np.arange(0, t_final, dt)

# Initial conditions
x0 = np.array([[0.5], [0.3]])
xr0 = np.array([[0.8], [-0.2]])

# Initialize arrays to store state and control input
x = np.zeros((2, len(t)))
xr = np.zeros((2, len(t)))
u = np.zeros(len(t))

# Initialize adaptive controller
controller = AdaptiveController()

# Simulation loop
for i in range(len(t)-1):
    # Reference model dynamics
    dxr = reference_model(xr[:, i], u[i])
    xr[:, i+1] = xr[:, i] + dxr * dt

    # Plant dynamics
    dx = plant_dynamics(x[:, i], u[i])
    x[:, i+1] = x[:, i] + dx * dt

    # Adaptive control law
    u[i] = controller.control_law(x[:, i], xr[:, i])

# Plotting the results
plt.figure()
plt.plot(t[:-1], xr[0, :-1], label='xr1')  # Remove the last element from t and xr using t[:-1] and xr[0, :-1]
plt.plot(t[:-1], xr[1, :-1], label='xr2')  # Remove the last element from t and xr using t[:-1] and xr[1, :-1]
plt.plot(t[:-1], x[0, :-1], label='x1')    # Remove the last element from t and x using t[:-1] and x[0, :-1]
plt.plot(t[:-1], x[1, :-1], label='x2')    # Remove the last element from t and x using t[:-1] and x[1, :-1]
plt.xlabel('Time')
plt.ylabel('States')
plt.legend()

plt.figure()
plt.plot(t[:-1], u[:-1])  # Remove the last element from t and u using t[:-1] and u[:-1]
plt.xlabel('Time')
plt.ylabel('Control Input')
plt.show()


