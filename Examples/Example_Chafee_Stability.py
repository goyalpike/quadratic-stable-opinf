import argparse
import os
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# import pandas as pd
from matplotlib import cm
from scipy.integrate import solve_ivp
from scipy.io import loadmat, savemat
from scipy.linalg import block_diag

from qs_opinf import module_models
from qs_opinf.constants import data_path, results_path
from qs_opinf.module_training import training
from qs_opinf.utils import ddt_uniform, reprod_seed

font = {"family": "normal", "weight": "bold", "size": 20}

matplotlib.rc("font", **font)
plt.rcParams["text.usetex"] = True

# activate latex text rendering
matplotlib.rc("text", usetex=True)
matplotlib.rc("axes", linewidth=1)
matplotlib.rc("font", weight="bold")
matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{sfmath} \boldmath"]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


reprod_seed(42)


@dataclass
class parameters:
    bs: int = 8
    num_epochs: int = 2000
    normalizing_coeffs: bool = False
    path: str | None = None
    model_hypothesis: str | None = None
    regularization_H: float = 0.0
    sys_order: int | None = None


Params = parameters()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_hypothesis",
    type=str,
    default="localstability",
    choices={"no_hypos", "localstability", "globalstability"},
    help="Enforcing model hypothesis",
)

parser.add_argument("--epochs", type=int, default=4000, help="Number of epochs")

args = parser.parse_args()
Params.num_epochs = args.epochs

path_funcs = {
    "no_hypos": (
        str(results_path / "Chafee/NoStability/"),
        module_models.ModelHypothesis,
        True,
    ),
    "localstability": (
        str(results_path / "Chafee/LocalStability/"),
        module_models.ModelHypothesisLocalStable,
        False,
    ),
    "globalstability": (
        str(results_path / "Chafee/GlobalStability/"),
        module_models.ModelHypothesisGlobalStable,
        False,
    ),
}

Params.path, model_hypothesis_function, B_term = path_funcs[args.model_hypothesis]

print(Params)
if not os.path.exists(Params.path):
    os.makedirs(Params.path)


# ## Loading data

data = loadmat(data_path / "Chafee_data_inits_conditions.mat")

X_all = data["X_data"].transpose(0, 2, 1)
x_shift = 1 * np.ones((1, 1000, 1))
X_all = X_all - x_shift

idxs = list(np.arange(0, 13))
testing_idxs = list([3, 7, 10])
train_idxs = list(set(idxs) - set(testing_idxs))


_X_testing = X_all[testing_idxs]
_X_training = X_all[train_idxs]

X_testing = np.concatenate([_X_testing, 0.5 * (0 * _X_testing + _X_testing**2)], axis=1)
X_training = np.concatenate([_X_training, 0.5 * (0 * _X_training + _X_training**2)], axis=1)

t = data["t"].T
print(f"Training trajectories: {X_training.shape}")
print(f"Testing trajectories: {X_testing.shape}")


# Make grid.
x = np.arange(0, 1, 1 / 1000)
y = np.array(t)
x, y = np.meshgrid(x, y)

fig, ax = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": "3d"})
# Plot the surface.
surf1 = ax[0].plot_surface(x, y, X_training[0][:1000].T, cmap=cm.coolwarm)
surf2 = ax[1].plot_surface(x, y, X_training[-1][:1000].T, cmap=cm.coolwarm)

# Preparing data for learning...
X_training_true = X_training
t_true = t

p = 2  # in training data, data are sampled at time-interval (p * dt)
X_training_sampled = X_training[..., ::p]
t = t[::p]

# Retain only the first k snapshots/inputs for training the ROM.
t_train = t  # Temporal domain for training snapshots.
X = X_training_sampled  # Training snapshots.

num_inits = X.shape[0]
print(f"shape of X for training:\t{X.shape}")
print(f"Training samples: {num_inits}")


# Compute SVD in order to prepare low-dimensional data
temp_X = X[0]

for i in range(1, X.shape[0]):
    temp_X = np.hstack((temp_X, X[i]))

[U, S, V] = np.linalg.svd(temp_X)

TOLS = np.logspace(np.log10(5e-2), -4, 8)
TOLS = [5e-2]
reduced_orders = []
for tol in TOLS:
    reprod_seed(42)
    font = {"family": "normal", "weight": "bold", "size": 20}

    matplotlib.rc("font", **font)
    plt.rcParams["text.usetex"] = True

    print("\n")

    [U1, S1, V1] = np.linalg.svd(temp_X[:1000])
    [U2, S2, V2] = np.linalg.svd(temp_X[1000:])
    r1 = 1
    while r1 < len(S1) + 1:
        if 1 - sum(S1[:r1]) / sum(S1) < tol:
            break
        r1 += 1

    print(f"Domainant model for first: {r1}")
    print(f"Energy captured by the snapshots: {100 * sum(S1[:r1]) / sum(S1)}%")

    r2 = 1
    while r2 < len(S2) + 1:
        if 1 - sum(S2[:r2]) / sum(S2) < tol:
            break
        r2 += 1

    Params.sys_order = r2
    print(f"Domainant model for second: {r2}")
    print(f"Energy captured by the snapshots: {100 * sum(S2[:r2]) / sum(S2)}%")

    Params.sys_order = r1 + r2
    r = Params.sys_order
    reduced_orders.append(r)

    print("\n")

    print(f"Order of reduced model: {r} and tolerance for svd is: {tol:.2e}")
    # print(f'Energy captured by the snapshots: {100*sum(S[:r])/sum(S)}%')
    # preparing reduced data, approximating derivative information, and dataloader for training
    Projection_V = block_diag(U1[:, :r1], U2[:, :r2])
    print(f"Proj matrix shape: {Projection_V.shape}")
    temp_Xr = Projection_V.T @ temp_X  # reduced data

    if Params.normalizing_coeffs:
        scaling_fac = np.max(np.abs(temp_Xr))
    else:
        scaling_fac = 1.0

    temp_Xr = temp_Xr / scaling_fac

    Xr = np.zeros((num_inits, r, temp_Xr.shape[-1] // num_inits))
    dXr = np.zeros((num_inits, r, temp_Xr.shape[-1] // num_inits))

    for i in range(0, num_inits):
        temp = int(temp_Xr.shape[-1] // num_inits)
        temp_x = temp_Xr[:, i * temp : (i + 1) * temp]
        temp_dx = ddt_uniform(temp_x, (t_train[1] - t_train[0]).item(), order=4)
        Xr[i] = temp_x
        dXr[i] = temp_dx

    # Define dataloaders
    t1 = (torch.arange(0, Xr.shape[-1]) * (t_train[1] - t_train[0])).reshape(-1, 1)

    train_dset = list(
        zip(
            torch.tensor(Xr).permute((0, 2, 1)).double(),
            torch.stack([t1 for _ in range(num_inits)], axis=0),
            torch.tensor(dXr).permute((0, 2, 1)).double(),
        )
    )
    train_dl = torch.utils.data.DataLoader(train_dset, batch_size=Params.bs, shuffle=True)
    dataloaders = {"train": train_dl}

    Params.regularization_H = 1e-8

    model = model_hypothesis_function(sys_order=r, B_term=B_term).double()  # type: ignore[attr-defined]

    opt_func = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        opt_func,
        step_size_up=2000 * len(dataloaders["train"]),
        mode="triangular2",
        cycle_momentum=False,
        base_lr=1e-6,
        max_lr=5e-3,
    )

    model, loss_track = training(model, dataloaders, opt_func, Params, scheduler=scheduler)

    ## Learned model
    A_OpInf = model.A.detach().numpy()
    H_OpInf = model.H.detach().numpy()
    B_OpInf = (
        model.B.detach()
        .numpy()
        .reshape(
            -1,
        )
    )

    def model_quad_OpInf(t, x):
        """Define a vector field of a quadratic system."""
        return A_OpInf @ x + H_OpInf @ np.kron(x, x) + B_OpInf

    # testing learned model
    x = np.arange(0, 1, 1 / 1000)
    y_testing = np.array(data["t"].T)
    x, y_testing = np.meshgrid(x, y_testing)
    PROPERTY = {
        "cmap": cm.viridis,
        "antialiased": False,
        "rstride": 10,
        "cstride": 10,
        "linewidth": 0,
    }

    Err_testing = []

    for k in range(X_testing.shape[0]):
        x0 = Projection_V.T @ X_testing[k, :, 0] / scaling_fac
        t_testing = np.arange(0, len(t_true)) * (t_true[1] - t_true[0])

        sol_OpInf = solve_ivp(model_quad_OpInf, [t_testing[0], t_testing[-1]], x0, t_eval=t_testing)
        full_sol_OpInf = Projection_V @ sol_OpInf.y

        if (sol_OpInf.y).shape[-1] == len(t_testing):
            fig, ax = plt.subplots(1, 3, figsize=(16, 4), subplot_kw={"projection": "3d"})
            # Plot the surface.
            surf = ax[0].plot_surface(
                x,
                y_testing,
                (X_testing[k][:1000] + x_shift.reshape(-1, 1)).T,
                **PROPERTY,
            )
            surf = ax[1].plot_surface(
                x,
                y_testing,
                scaling_fac * (full_sol_OpInf[:1000] + x_shift.reshape(-1, 1)).T,
                **PROPERTY,
            )
            surf = ax[2].plot_surface(
                x,
                y_testing,
                np.log10(abs(X_testing[k][:1000].T - scaling_fac * full_sol_OpInf[:1000].T)),
                **PROPERTY,
            )
            ax[0].set(xlabel="$x$", ylabel="time", zlabel="$u(x,t)$", title="ground-truth")
            ax[1].set(
                xlabel="$x$",
                ylabel="time",
                zlabel="$\hat{u}(x,t)$",
                title="learned model",
            )
            ax[2].set(
                xlabel="$x$",
                ylabel="time",
                zlabel="error in log-scale",
                title="absolute error",
            )

            ax[0].set_zlim([0, 1.25])
            ax[1].set_zlim([0, 1.25])

            for _ax in ax:
                _ax.xaxis.labelpad = 10
                _ax.yaxis.labelpad = 10
                _ax.zaxis.labelpad = 10

            ax[2].tick_params(axis="z", direction="out", pad=10)
            ax[2].zaxis.labelpad = 20

            ax[2].set_zscale("linear")

            plt.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.1)

            # plt.show()
            fig.savefig(
                Params.path + f"simulation_test_{k}_order_{r}.pdf",
                bbox_inches="tight",
                pad_inches=0,
            )
            fig.savefig(
                Params.path + f"simulation_test_{k}_order_{r}.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )

            err = (np.linalg.norm(scaling_fac * full_sol_OpInf - X_testing[k])) / (
                np.linalg.norm(X_testing[k])
            )
            Err_testing.append(err)

        else:
            Err_testing.append(np.NaN)
    print(f"error: {Err_testing}!\n")
    Errors = np.mean(Err_testing)
    print(f"Errors: {Errors} | Err_testing: {Err_testing}!")
    savemat(
        Params.path + f"simulation_error_order_{r}.mat",
        {
            "errors": Errors,
            "loss": loss_track,
            "eigs": np.linalg.eig(A_OpInf)[0],
            "reduced_orders": reduced_orders,
            "sin_vals1": S1,
            "sin_vals2": S2,
        },
    )
    print(f"max eigenvalue: {np.max(np.real(np.linalg.eig(A_OpInf)[0]))}")
    plt.close("all")
