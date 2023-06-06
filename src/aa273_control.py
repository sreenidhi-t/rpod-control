# %%
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


# %%

# ------- DYNAMICS -------- #
def state_space(T, n, m):
    """CWH State Space Representation"""

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

def dynamics(dt, N, mean_motion, mass):
    """Returns A and B matrices for each time step"""
    A = []
    B = []

    for k in range(N):
        # print("hi")
        Anew, Bnew = state_space(k*dt, mean_motion, mass)
        A.append(Anew)
        B.append(Bnew)
        
    return (A, B)

# %%
def straight_line_traj(s_start, s_goal, T):
    n = s_start.shape[0]
    s = np.zeros((T + 1, n))
    for k in range(T + 1):
        s[k] = s_start + (s_goal - s_start)*k/T
    sx = s[:, 0]
    sy = s[:, 1]
    sz = s[:, 2]
    vx = np.diff(sx)
    vy = np.diff(sy)
    vz = np.diff(sz)
    vx = np.append(vx, vx[-1])
    vy = np.append(vy, vy[-1])
    vz = np.append(vz, vz[-1])

    s = np.hstack((sx[:, None], sy[:, None], sz[:, None], vx[:, None], vy[:, None], vz[:, None]))
    return s

# %%
def do_MPC(dt, chaser_n, chaser_m, s_current, s_goal, N, Q, R, P, max_iters, eps):
    """Performs MPC for one time step and calculates the optimal control input at that time
    Returns:
        s_mpc: state trajectory for time horizon N
        u_mpc: control input for time horizon N

    Inputs:
        s0: initial state at current time step (6x1) #TODO: CHECK DIMENSIONS
        s_goal: goal state (6x1)
        N: MPC time horizon
        Q: state cost matrix
        R: control cost matrix
        P: terminal state cost matrix
        max_iters: maximum number of iterations for SCP
        eps: convergence threshold for SCP
    """
    
    n = Q.shape[0]
    m = R.shape[0]

    #TODO: SEE WHAT 273 CODE USES FOR A AND B STATE SPACE MATRICES
    A, B = dynamics(dt, N, chaser_n, chaser_m)

    # Generate a straight line trajectory from s_start to s_goal for MPC Horizon length
    # This is currently not working B)!
    # s_init = straight_line_traj(s_current, s_goal, N)

    # Construct the convex SCP sub-problem.
    s_cvx = cp.Variable((N + 1, n))
    u_cvx = cp.Variable((N, m))
    s0 = s_current

    objective = cp.quad_form((s_cvx[N] - s_goal), P) + cp.sum([(cp.quad_form((s_cvx[k] - s_goal), Q) + cp.quad_form(u_cvx[k], R)) for k in range(N)])
    constraints = [s_cvx[0] == s0]
    constraints += [(s_cvx[k + 1] == A[k]@s_cvx[k] + B[k]@u_cvx[k]) for k in range(N)]
    prob = cp.Problem(cp.Minimize(objective), constraints)

    # ------- SCP Solve-------- #
    # Solve SCP until convergence or max iterations
    converged = False
    J = np.zeros(max_iters + 1)
    J[0] = np.inf
    count = 0
    for i in range(max_iters):
        # prob.solve(warm_start=True)
        prob.solve()
        s, u, J[i + 1] = s_cvx.value, u_cvx.value, prob.objective.value

        dJ = np.abs(J[i + 1] - J[i])
        count += 1
        if dJ < eps:
            converged = True
            break
    
    print(f"Converged in {count} iterations")
    if not converged:
        raise RuntimeError('SCP did not converge!')

    return s, u




