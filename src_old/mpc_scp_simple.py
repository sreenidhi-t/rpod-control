from functools import partial

import cvxpy as cp

# import jax
# import jax.numpy as jnp

import matplotlib.pyplot as plt

import numpy as np



# ------- CONSTANTS ------- #

n = 6 # state dimension
m = 3 # control dimension
T = 50 # total time
N = 30 # MPC horizon
dt = 1
mass = 1.0 # Mass
mean_motion = 1
rho = 20
max_iters = 50 # maximum SCP iterations
eps = 0.5 # convergence tolerance
u_max = 1000 # control effort bound

P = 1e3*np.eye(n)                    # terminal state cost matrix
Q = np.diag([1, 1., 1, 1e-3, 1e-3, 1e-3])
# Q = np.diag([1e-1, 1e-1, 1e-1, 1e-2, 1e-2, 1e-2])  # state cost matrix
R = 1e-3*np.eye(m)  


s_start = np.array([50, 50, 50, 0, 0, 0]) # x, y, z, dx, dy, dz
s_goal = np.array([0, 0, 0, 0, 0, 0]) # location of target in space


# ------- DYNAMICS -------- #

def state_space(T, n, m):
    # CWH equations for state space
    A = np.matrix([[4 - 3*np.cos(n*T), 0, 0, 1/n*np.sin(n*T), 2/n*(1-np.cos(n*T)), 0],
                  [6*(np.sin(n*T)-n*T), 1, 0, 2/n*(np.cos(n*T)-1), 1/n*(4*np.sin(n*T)-3*n*T), 0],
                  [0, 0, np.cos(n*T), 0, 0, 1/n*np.sin(n*T)],
                  [3*n*np.sin(n*T), 0, 0, np.cos(n*T), 2*np.sin(n*T), 0],
                  [6*n*(np.cos(n*T)-1), 0, 0, -2*np.sin(n*T), 4*np.cos(n*T)-3, 0],
                  [0, 0, -n*np.sin(n*T), 0, 0, np.cos(n*T)]])
    
    B = 1/m*np.matrix([[1/n**2*(1-np.cos(n*T)), 2/n**2*(n*T - np.sin(n*T)), 0],
                     [-2/n**2*(n*T - np.sin(n*T)), 4/n**2*(1-np.cos(n*T))-3/n*T, 0],
                     [0, 0, 1/n**2*(1-np.cos(n*T))],
                     [1/n*np.sin(n*T), -2/n*(np.cos(n*T)-1), 0],
                     [2/n*(np.cos(n*T)-1), 4/n*np.sin(n*T)-3/n*T, 0],
                     [0, 0, 1/n*np.sin(n*T)]])

    return (A, B)

def dynamics(N, mean_motion, mass):
    
    A = []
    B = []

    for k in range(N):
        # print("hi")
        Anew, Bnew = state_space(k, mean_motion, mass)
        A.append(Anew)
        B.append(Bnew)
        
    return (A, B)


# ------- SCP Problem Definition -------- #
A, B = dynamics(T, mean_motion, mass)

n = Q.shape[0]
m = R.shape[0]

s_cvx = cp.Parameter((N + 1, n))
u_cvx = cp.Parameter((N, m))
s0 = cp.Parameter(n)
s0.value = s_start

# Construct the convex SCP sub-problem.
objective = [cp.quad_form((s_cvx[-1] - s_goal), P)]
objective += [cp.sum(cp.quad_form((s_cvx[k] - s_goal), Q) + cp.quad_form(u_cvx[k], R)) for k in range(N)]
objective = cp.sum(objective)

constraints = [s_cvx[0] == s0,
                cp.norm(u_cvx, 'inf') <= u_max]
constraints += [(s_cvx[k + 1] == A[k]@s_cvx[k] + B[k]@u_cvx[k]) for k in range(N)]

prob = cp.Problem(cp.Minimize(objective), constraints)

# ------- SCP Iteration -------- #
def scp_iteration(s_new, u_new):
    s_cvx.value = s_new
    u_cvx.value = u_new
    prob.solve()

    return s_cvx.value, u_cvx.value, prob.objective.value

# ------- SCP Solve-------- #
def scp_solve(max_iters, eps, s_init, u_init):

    if s_init is None or u_init is None:
        s = np.zeros((N + 1, n))
        u = np.zeros((N, m))
        s[0] = np.array(s0.value).flatten()
        for k in range(N):
            s[k+1] = A[k]@s[k] + B[k]@u[k]
    else:
        s = np.copy(s_init)
        u = np.copy(u_init)

    converged = False
    J = np.zeros(max_iters + 1)
    J[0] = np.inf
    

    count = 0
    for i in range(max_iters):
        s, u, J[i + 1] = scp_iteration(s, u)
        dJ = np.abs(J[i + 1] - J[i])
        count += 1
        if dJ < eps:
            converged = True
            break
    print(count)
    if not converged:
        raise RuntimeError('SCP did not converge!')
    return s, u



# ------- MPC -------- #
s_mpc = np.zeros((T, N + 1, n))
u_mpc = np.zeros((T, N, m))

for t in range(T):
    # ------- INITIALIZE STRAIGHT LINE TRAJECTORY -------- #
    # convert to spherical coordinates
    rho = np.sqrt(s0.value[0]**2 + s0.value[1]**2 + s0.value[2]**2)
    theta = np.arccos(s_start[2]/rho)
    phi = np.arctan2(s_start[1], s_start[0])

    # create straight line trajectory
    r_final = np.sqrt(s_goal[0]**2 + s_goal[1]**2 + s_goal[2]**2)
    
    rho_traj = np.linspace(rho, r_final, N + 1)
    
    sx_traj = [rho_traj[k]*np.sin(theta)*np.cos(phi) for k in range(N + 1)]
    sy_traj = [rho_traj[k]*np.sin(theta)*np.sin(phi) for k in range(N + 1)]
    sz_traj = [rho_traj[k]*np.cos(theta) for k in range(N + 1)]

    vx_traj = [(sx_traj[k+1] - sx_traj[k]) for k in range(N)] + [0]
    vy_traj = [(sy_traj[k+1] - sy_traj[k]) for k in range(N)] + [0]
    vz_traj = [(sz_traj[k+1] - sz_traj[k]) for k in range(N)] + [0]

    s_init = np.array([sx_traj, sy_traj, sz_traj, vx_traj, vy_traj, vz_traj]).transpose()
    u_init = np.zeros((N, m))
    print(s_init.shape)
    print(s_cvx.shape)

    s_mpc[t], u_mpc[t] = scp_solve(max_iters, eps, s_init, u_init)
    
    status = prob.status
    
    if status == 'infeasible':
        s_mpc = s_mpc[:t]
        u_mpc = u_mpc[:t]
        print('MPC problem is infeasible!')
        break 

    s0.value = np.array(A[t]@s0.value + B[t]@u_mpc[t, 0, :]).flatten()

    # plt.plot(s_mpc[t, :, 0], s_mpc[t, :, 1], '--*', color='k')

print(s_mpc.shape)

plt.plot(s_mpc[:, 0, 0], s_mpc[:, 0, 1], '-o')
# plot starting point and goal
plt.plot(s_mpc[0, 0, 0], s_mpc[0, 0, 1], 'o', color='r')
plt.plot(s_mpc[-1, 0, 0], s_mpc[-1, 0, 1], 'o', color='g')
plt.show()