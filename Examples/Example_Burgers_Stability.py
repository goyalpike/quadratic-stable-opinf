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

from qs_opinf import module_models
from qs_opinf.module_training import training
from qs_opinf.utils import ddt_uniform, reprod_seed
from qs_opinf.constants import data_path

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
    path: str = None
    model_hypothesis: str = None


Params = parameters()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_hypothesis",
    type=str,
    default="no_hypos",
    choices={"no_hypos", "localstability", "globalstability"},
    help="Enforcing model hypothesis",
)

parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs")

args = parser.parse_args()
Params.num_epochs = args.epochs

path_funcs = {
    "no_hypos": ("./Results/Burgers/NoStability/", module_models.ModelHypothesis, True),
    "localstability": (
        "./Results/Burgers/LocalStability/",
        module_models.ModelHypothesisLocalStable,
        False,
    ),
    "globalstability": (
        "./Results/Burgers/GlobalStability/",
        module_models.ModelHypothesisGlobalStable,
        False,
    ),
}

Params.path, model_hypothesis_function, B_term = path_funcs[args.model_hypothesis]

print(Params)
if not os.path.exists(Params.path):
    os.makedirs(Params.path)


# ## Loading data

data = loadmat("Burger_data_init_condions.mat")
A = data["Af"]
H = data["H"]

X = data["xo"].T
X_all = data["X_data"].transpose(0, 2, 1)
idxs = list(np.arange(0, 17))
testing_idxs = list([4, 8, 12])
train_idxs = list(set(idxs) - set(testing_idxs))

X_testing = X_all[testing_idxs]
X_training = X_all[train_idxs]

t = data["t"].T
print(f"Training trajectories: {X_training.shape[0]}")
print(f"Training trajectories: {X_testing.shape[0]}")


# Make grid.
x = np.arange(0, 1, 1 / 1000)
y = np.array(t)
x, y = np.meshgrid(x, y)

fig, ax = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": "3d"})
# Plot the surface.
surf1 = ax[0].plot_surface(x, y, X_training[0].T, cmap=cm.coolwarm)
surf2 = ax[1].plot_surface(x, y, X_training[-1].T, cmap=cm.coolwarm)

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

fig, ax = plt.subplots(1, 1)
plt.semilogy(S)
plt.grid()
ax.legend([])


reduced_orders = np.arange(12, 29, 2)
print(f"reduced_order are {reduced_orders}!")

for r in reduced_orders:
    font = {"family": "normal", "weight": "bold", "size": 18}

    matplotlib.rc("font", **font)
    plt.rcParams["text.usetex"] = True

    print("\n")

    print(f"Order of reduced model: {r}")
    print(f"Energy captured by the snapshots: {100*sum(S[:r])/sum(S)}%")
    # preparing reduced data, approximating derivative information, and dataloader for training
    temp_Xr = U[:, :r].T @ temp_X  # reduced data

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
    train_dl = torch.utils.data.DataLoader(
        train_dset, batch_size=Params.bs, shuffle=True
    )
    dataloaders = {"train": train_dl}

    Params.regularization_H = 1e-8

    model = model_hypothesis_function(sys_order=r, B_term=B_term).double()

    opt_func = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        opt_func,
        step_size_up=2000 * len(dataloaders["train"]),
        mode="triangular",
        cycle_momentum=False,
        base_lr=1e-6,
        max_lr=5e-3,
    )

    model, loss_track = training(
        model, dataloaders, opt_func, Params, scheduler=scheduler
    )

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
        return A_OpInf @ x + H_OpInf @ np.kron(x, x) + B_OpInf

    x = np.arange(0, 1, 1 / 1000)
    y_testing = np.array(data["t"].T)
    x, y_testing = np.meshgrid(x, y_testing)
    Err_testing = []

    # testing of the learned model
    for k in range(X_testing.shape[0]):

        x0 = U[:, :r].T @ X_testing[k, :, 0] / scaling_fac
        t_testing = np.arange(0, 501) * (t_true[1] - t_true[0])

        sol_OpInf = solve_ivp(
            model_quad_OpInf, [t_testing[0], t_testing[-1]], x0, t_eval=t_testing
        )
        full_sol_OpInf = U[:, :r] @ sol_OpInf.y

        if (sol_OpInf.y).shape[-1] == len(t_testing):
            fig, ax = plt.subplots(
                1, 3, figsize=(16, 4), subplot_kw={"projection": "3d"}
            )
            # Plot the surface.
            surf = ax[0].plot_surface(x, y_testing, X_testing[k].T, cmap=cm.coolwarm)
            surf = ax[1].plot_surface(
                x, y_testing, scaling_fac * full_sol_OpInf.T, cmap=cm.coolwarm
            )
            surf = ax[2].plot_surface(
                x,
                y_testing,
                np.log10(abs(X_testing[k].T - scaling_fac * full_sol_OpInf.T)),
                cmap=cm.coolwarm,
            )
            ax[0].set(
                xlabel="$x$", ylabel="time", zlabel="$u(x,t)$", title="ground-truth"
            )
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

            ax[0].set_zlim([0, 2])
            ax[1].set_zlim([0, 2])

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

    Errors = np.mean(Err_testing)
    savemat(
        Params.path + f"simulation_error_order_{r}.mat",
        {
            "errors": Errors,
            "loss": loss_track,
            "eigs": np.linalg.eig(A_OpInf)[0],
            "reduced_orders": reduced_orders,
            "sin_vals": S,
        },
    )

    plt.close("all")
