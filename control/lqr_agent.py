import numpy as np
from scipy.linalg import solve_continuous_are as sol_riccatti
from copy import deepcopy

class LQRAgent():
    def __init__(self, env, sim_env):
        self.env = env
        self.sim_env = sim_env
    
    def simulate_dynamics(self, x, u):
        self.sim_env.state = x.copy()
        x_1 = np.array(self.sim_env.step(u)[0])
        return x_1-x.copy()

    def approximate_A(self, x, u, delta=1e-5):
        A = np.zeros((x.shape[0], x.shape[0]))
        for i in range(A.shape[1]):

            x_perturbed = x.copy()
            x_perturbed[i] -= delta

            pert_neg = self.simulate_dynamics(x_perturbed, u)

            x_perturbed = x.copy()
            x_perturbed[i] += delta
            pert_pos = self.simulate_dynamics(x_perturbed, u)

            A[:,i] = (pert_pos-pert_neg) / (2*delta)

        return A


    def approximate_B(self, x, u, delta=1e-5):
        B = np.zeros((x.shape[0], u.shape[0]))
        for i in range(B.shape[1]):

            u_perturbed = u.copy()
            u_perturbed[i] -= delta
            pert_neg = self.simulate_dynamics(x, u_perturbed)

            u_perturbed = u.copy()
            u_perturbed[i] += delta
            pert_pos = self.simulate_dynamics(x, u_perturbed)

            B[:,i] = (pert_pos-pert_neg) / (2*delta)

        return B

    def action(self, goal=None):
        if goal == None:
            goal = np.zeros(self.env.state.shape)
        u = np.zeros(self.env.action_space.shape)
        x = self.env.state
        A = self.approximate_A(x, u)
        B = self.approximate_B(x, u)
        Q = np.eye(x.shape[0])
        R = np.eye(self.env.action_space.shape[0])
        P = sol_riccatti(A, B, Q, R)
        K = np.linalg.inv(R) @ B.T @ P
        u = -K @ (x-np.copy(goal))
        return u