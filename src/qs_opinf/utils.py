#!/usr/bin/env python3

import numpy as np
import scipy.linalg as la
import torch


def reprod_seed(random_seed):
    """Set seed for reproducibility."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


def implicit_euler(t, q0, A, B, U):
    """Solve the system
        dq / dt = Aq(t) + Bu(t),    q(0) = q0,
    over a uniform time domain via the implicit Euler method.

    Parameters
    ----------
    t : (k,) ndarray
        Uniform time array over which to solve the ODE.
    q0 : (n,) ndarray
        Initial condition.
    A : (n, n) ndarray
        State matrix.
    B : (n,) or (n, 1) ndarray
        Input matrix.
    U : (k,) ndarray
        Inputs over the time array.

    Returns
    -------
    q : (n, k) ndarray
        Solution to the ODE at time t; that is, q[:,j] is the
        computed solution corresponding to time t[j].

    """
    # Check and store dimensions.
    k = len(t)
    n = len(q0)
    B = np.ravel(B)
    assert A.shape == (n, n)
    assert B.shape == (n,)
    assert U.shape == (k,)
    I = np.eye(n)

    # Check that the time step is uniform.
    dt = t[1] - t[0]
    assert np.allclose(np.diff(t), dt)

    # Factor I - dt*A for quick solving at each time step.
    factored = la.lu_factor(I - dt * A)

    # Solve the problem at each time step.
    q = np.empty((n, k))
    q[:, 0] = q0.copy()
    for j in range(1, k):
        q[:, j] = la.lu_solve(factored, q[:, j - 1] + dt * B * U[j])

    return q


def kron(x, y):
    """Define kronecker product for row-wise vectors."""
    return torch.einsum("ab,ad->abd", [x, y]).view(x.size(0), x.size(1) * y.size(1))


## Simple RK model
def rk4th_onestep(model, x, t=0, timestep=1e-2):
    """Define Runge-Kutta scheme to predict state vector at the next time step."""
    k1 = model(x, t)
    k2 = model(x + 0.5 * timestep * k1, t + 0.5 * timestep)
    k3 = model(x + 0.5 * timestep * k2, t + 0.5 * timestep)
    k4 = model(x + 1.0 * timestep * k3, t + 1.0 * timestep)
    return x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * timestep


# def training(model, dataloaders, opt_func, Params, p=1.0):
#     """Given model, dataloaders and optimizer, it optimizes models/parameters.

#     Args:
#         model: Neural network model
#         dataloaders: Training data
#         opt_func: Optimizer
#         Params: Parameters (e.g., number of epochs)
#         scheduler (optional): Learning rate sheduler. Defaults to None.

#     Returns
#     -------
#         (model, loss): Trained model and loss wrt epoch.
#     """
#     define_selectionVector = True

#     criteria = nn.MSELoss()
#     loss_track = []
#     for g in range(Params.num_epochs):
#         for y in dataloaders["train"]:
#             opt_func.zero_grad()
#             total_loss = 0.0
#             for i in range(y[0].shape[0]):
#                 yi = y[0][i]
#                 timesteps_i = torch.tensor(np.diff(y[1][i], axis=0)).float()
#                 y_total = yi

#                 ##################################
#                 # One forward step predictions
#                 ##################################
#                 y_pred = rk4th_onestep(model, y_total[:-1], timestep=timesteps_i)

#                 if define_selectionVector:
#                     selectionVector = torch.tensor(
#                         np.random.choice([1, 0], size=(len(y_pred),), p=[p, 1 - p])
#                     ).reshape(-1, 1)
#                     # define_selectionVector = False
#                     # print('Define selection vector!!!')

#                 total_loss += (1 / timesteps_i.mean()) * criteria(
#                     selectionVector * y_pred, selectionVector * y_total[1:]
#                 )

#                 ##################################
#                 # One backward step predictions
#                 ##################################
#                 y_pred_back = rk4th_onestep(model, y_total[1:], timestep=-timesteps_i)
#                 total_loss += (
#                     0
#                     * (1 / timesteps_i.mean())
#                     * criteria(selectionVector * y_pred_back, selectionVector * y_total[:-1])
#                 )

#             total_loss /= y[0].shape[0]
#             loss_track.append(total_loss.item())
#             total_loss.backward()
#             opt_func.step()

#         sys.stdout.write(
#             "\r [Epoch %d/%d] [Training loss: %.2e] [Learning rate: %.2e]"
#             % (g + 1, Params.num_epochs, loss_track[g], opt_func.param_groups[0]["lr"])
#         )

#         if g == Params.num_epochs // 2:
#             opt_func = torch.optim.Adam(
#                 model.parameters(), lr=Params.lr / 10, weight_decay=Params.weightdecay
#             )

#     print(1 / timesteps_i.mean())

#     return model, loss_track


# def training_deri_info(
#     model,
#     dataloaders,
#     opt_func,
#     Params,
#     p=1.0,
#     include_deriInfo=True,
#     RK_Inegrator=True,
# ):
#     if include_deriInfo:
#         print("Including derivative information in learning!")
#     else:
#         print("Do NOT include derivative information in learning!")

#     if RK_Inegrator:
#         print("Imposing RK Integrator!")
#     else:
#         print("NOT imposing RK Integrator!")

#     if not (include_deriInfo or RK_Inegrator):
#         raise ValueError(
#             "Both RK_Inegrator and include_deriInfo cannot be negative at the same time!"
#         )

#     define_selectionVector = True

#     criteria = nn.MSELoss()
#     loss_track = []
#     for g in range(Params.num_epochs):
#         for y in dataloaders["train"]:
#             opt_func.zero_grad()
#             total_loss = 0.0
#             for i in range(y[0].shape[0]):
#                 yi = y[0][i]
#                 timesteps_i = torch.tensor(np.diff(y[1][i], axis=0)).float()
#                 y_total = yi

#                 if RK_Inegrator:
#                     ##################################
#                     # One forward step predictions
#                     ##################################
#                     y_pred = rk4th_onestep(model, y_total[:-1], timestep=timesteps_i)

#                     if define_selectionVector:
#                         selectionVector = torch.tensor(
#                             np.random.choice([1, 0], size=(len(y_pred),), p=[p, 1 - p])
#                         ).reshape(-1, 1)
#                         # define_selectionVector = False
#                         # print('Define selection vector!!!')

#                     total_loss += (1 / timesteps_i.mean()) * criteria(
#                         selectionVector * y_pred, selectionVector * y_total[1:]
#                     )

#                     ##################################
#                     # One backward step predictions
#                     ##################################
#                     y_pred_back = rk4th_onestep(model, y_total[1:], timestep=-timesteps_i)
#                     total_loss += (
#                         0
#                         * (1 / timesteps_i.mean())
#                         * criteria(
#                             selectionVector * y_pred_back,
#                             selectionVector * y_total[:-1],
#                         )
#                     )

#                 if include_deriInfo:
#                     ##################################
#                     # Compute derivatives
#                     deri_info = model(y_total, 0)
#                     loss_deri = criteria(deri_info[2:-2, :], y[2][i][2:-2, :])
#                     total_loss += loss_deri

#                 ##################################

#             total_loss /= y[0].shape[0]
#             loss_track.append(total_loss.item())
#             total_loss.backward()
#             opt_func.step()

#         sys.stdout.write(
#             "\r [Epoch %d/%d] [Training loss: %.2e] [Learning rate: %.2e]"
#             % (g + 1, Params.num_epochs, loss_track[g], opt_func.param_groups[0]["lr"])
#         )

#         if g == Params.num_epochs // 2:
#             opt_func = torch.optim.Adam(
#                 model.parameters(), lr=Params.lr / 10, weight_decay=Params.weightdecay
#             )

#     print(1 / timesteps_i.mean())

#     return model, loss_track


def ddt_uniform(states, dt, order=2):
    """Approximate the time derivatives for a chunk of snapshots that are
    uniformly spaced in time.

    Parameters
    ----------
    states : (n, k) ndarray
        States to estimate the derivative of. The jth column is a snapshot
        that corresponds to the jth time step, i.e., states[:, j] = x(t[j]).
    dt : float
        The time step between the snapshots, i.e., t[j+1] - t[j] = dt.
    order : int {2, 4, 6}
        The order of the derivative approximation.
        See https://en.wikipedia.org/wiki/Finite_difference_coefficient.

    Returns
    -------
    ddts : (n, k) ndarray
        Approximate time derivative of the snapshot data. The jth column is
        the derivative dx / dt corresponding to the jth snapshot, states[:, j].

    """
    # Check dimensions and input types.
    if states.ndim != 2:
        raise ValueError("states must be two-dimensional")
    if not np.isscalar(dt):
        raise TypeError("time step dt must be a scalar (e.g., float)")

    if order == 2:
        return np.gradient(states, dt, edge_order=2, axis=1)

    Q = states
    ddts = np.empty_like(states)
    n, k = states.shape
    if order == 4:
        # Central difference on interior.
        ddts[:, 2:-2] = (Q[:, :-4] - 8 * Q[:, 1:-3] + 8 * Q[:, 3:-1] - Q[:, 4:]) / (
            12 * dt
        )

        # Forward / backward differences on the front / end.
        for j in range(2):
            ddts[:, j] = _fwd4(Q[:, j : j + 5].T, dt)  # Forward
            ddts[:, -j - 1] = -_fwd4(Q[:, -j - 5 : k - j].T[::-1], dt)  # Backward

    elif order == 6:
        # Central difference on interior.
        ddts[:, 3:-3] = (
            -Q[:, :-6]
            + 9 * Q[:, 1:-5]
            - 45 * Q[:, 2:-4]
            + 45 * Q[:, 4:-2]
            - 9 * Q[:, 5:-1]
            + Q[:, 6:]
        ) / (60 * dt)

        # Forward / backward differences on the front / end.
        for j in range(3):
            ddts[:, j] = _fwd6(Q[:, j : j + 7].T, dt)  # Forward
            ddts[:, -j - 1] = -_fwd6(Q[:, -j - 7 : k - j].T[::-1], dt)  # Backward

    else:
        raise NotImplementedError(
            f"invalid order '{order}'; valid options: {{2, 4, 6}}"
        )

    return ddts


# Finite difference stencils ==================================================
def _fwd4(y, dt):
    """Compute the first derivative of a uniformly-spaced-in-time array with a
    fourth-order forward difference scheme.

    Parameters
    ----------
    y : (5, ...) ndarray
        Data to differentiate. The derivative is taken along the first axis.
    dt : float
        Time step (the uniform spacing).

    Returns
    -------
    dy0 : float or (...) ndarray
        Approximate derivative of y at the first entry, i.e., dy[0] / dt.

    """
    return (-25 * y[0] + 48 * y[1] - 36 * y[2] + 16 * y[3] - 3 * y[4]) / (12 * dt)


def _fwd6(y, dt):
    """Compute the first derivative of a uniformly-spaced-in-time array with a
    sixth-order forward difference scheme.

    Parameters
    ----------
    y : (7, ...) ndarray
        Data to differentiate. The derivative is taken along the first axis.
    dt : float
        Time step (the uniform spacing).

    Returns
    -------
    dy0 : float or (...) ndarray
        Approximate derivative of y at the first entry, i.e., dy[0] / dt.

    """
    return (
        -147 * y[0]
        + 360 * y[1]
        - 450 * y[2]
        + 400 * y[3]
        - 225 * y[4]
        + 72 * y[5]
        - 10 * y[6]
    ) / (60 * dt)
