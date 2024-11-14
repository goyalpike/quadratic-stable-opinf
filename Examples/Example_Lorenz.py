import argparse
import os
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp

from qs_opinf import module_models
from qs_opinf.module_training import training
from qs_opinf.utils import ddt_uniform, reprod_seed

font = {"family": "normal", "weight": "bold", "size": 18}
matplotlib.rc("font", **font)
plt.rcParams["text.usetex"] = True
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

reprod_seed(42)


@dataclass
class parameters:
    bs: int = 1
    num_epochs: int = 2000
    normalizing_coeffs: bool = False
    path: str | None = None
    model_hypothesis: str | None = None
    regularization_H: float = 0.0


Params = parameters()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_hypothesis",
    type=str,
    default="no_hypos",
    choices={"no_hypos", "globalstability"},
    help="Enforcing model hypothesis",
)
parser.add_argument(
    "--comparison_plots", action="store_true", help="Plotting eigen-value comparisons"
)

parser.add_argument("--epochs", type=int, default=12000, help="Number of epochs")

args = parser.parse_args()
Params.num_epochs = args.epochs

path_funcs = {
    "no_hypos": (
        "./Results/Lorenz/NoStability/",
        module_models.SINDy_NoStable,
    ),
    "globalstability": (
        "./Results/Lorenz/GlobalStability/",
        module_models.SINDy_Stable_Trapping,
    ),
}

Params.path, model_hypothesis_function = path_funcs[args.model_hypothesis]

print(Params)
if not os.path.exists(Params.path):
    os.makedirs(Params.path)


# Defining model
def lorenz_system(t, Y, sigma, rho, beta):
    """Define Lorenz vector field."""
    x, y, z = Y
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def plotting_training_data():
    """Plot training data."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sol.y[0][::2] / 8, sol.y[1][::2] / 8, (sol.y[2][::2] - 25) / 8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()

    fig.savefig(Params.path + "Noisy_measurement.pdf", bbox_inches="tight")
    fig.savefig(Params.path + "Noisy_measurement.png", dpi=300, bbox_inches="tight")


sigma, rho, beta = 10, 28, 8 / 3  # model parameters

t_train = np.linspace(0, 20, 10000)  # time span

Y0 = [-8.0, 7.0, 27.0]  # initial condition

# Generating data
sol = solve_ivp(
    fun=lambda t, Y: lorenz_system(t, Y, sigma, rho, beta),
    t_span=[t_train[0], t_train[-1]],
    y0=Y0,
    t_eval=t_train,
)

sol.y = sol.y + 10e-2 * np.random.randn(
    *sol.y.shape
)  # adding artificial Gaussian noise
plotting_training_data()

# Normalzing data
sol.y[0] = sol.y[0] / 8
sol.y[1] = sol.y[1] / 8
sol.y[2] = (sol.y[2] - 25) / 8

##################################################################################
# Preparing data for learning and creating dataloaders
X_training = sol.y.reshape(1, 3, -1)
t = t_train
X_training_true = X_training
t_true = t

p = 2  # in training data, data are sampled at time-interval (p * dt)
X_training_sampled = X_training[..., ::p]
t = t[::p]
t_train = t  # Temporal domain for training snapshots.

X = X_training_sampled  # Training snapshots.

num_inits = X.shape[0]
print(f"shape of X for training:\t{X.shape}")
print(f"Training samples: {num_inits}")

temp_X = X[0]

for i in range(1, X.shape[0]):
    temp_X = np.hstack((temp_X, X[i]))

[U, S, V] = np.linalg.svd(temp_X)


r = 3
temp_Xr = temp_X  # reduced data (here it is original data only)
if Params.normalizing_coeffs:
    scaling_fac = np.max(np.abs(temp_Xr))
else:
    scaling_fac = 1.0
print("scaling fac is: ", scaling_fac)

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

##################################################################################
# Defining mode, optimizer, and inference
Params.regularization_H = 1e-6  # regularizer for H
model = model_hypothesis_function(sys_order=r, B_term=True).double()  # type: ignore[attr-defined]
opt_func = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.CyclicLR(
    opt_func,
    step_size_up=2000 * len(dataloaders["train"]),
    mode="triangular",
    cycle_momentum=False,
    base_lr=1e-6,
    max_lr=1e-3,
)

model, loss_track = training(model, dataloaders, opt_func, Params, scheduler=scheduler)


## Hard-pruning, meaning removing small coefficients
TOL = 1e-1
for _ in range(4):
    # Removing the coefficients smaller than tol and set gradients w.r.t. them to zero
    # so that they will not be updated in the iterations
    Ws = model._J.detach().clone()
    Mask_Ws = (Ws.abs() > TOL).type(torch.float)
    model._J = torch.nn.Parameter(Ws * Mask_Ws)
    model._J.register_hook(lambda grad: grad.mul_(Mask_Ws))

    Wh = model._H_tensor.detach().clone()
    Mask_Wh = (Wh.abs() > TOL).type(torch.float)
    model._H_tensor = torch.nn.Parameter(Wh * Mask_Wh)
    model._H_tensor.register_hook(lambda grad: grad.mul_(Mask_Wh))

    Wb = model.B.detach().clone()
    Mask_Wb = (Wb.abs() > TOL).type(torch.float)
    model.B = torch.nn.Parameter(Wb * Mask_Wb)
    model.B.register_hook(lambda grad: grad.mul_(Mask_Wb))

    Wm = model.m.detach().clone()
    Mask_Wm = (Wm.abs() > TOL).type(torch.float)
    model.m = torch.nn.Parameter(Wm * Mask_Wm)
    model.m.register_hook(lambda grad: grad.mul_(Mask_Wm))

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
    print("\nA_opInf:", A_OpInf)
    print("\nH_opInf:", H_OpInf)
    print("\nB_opInf:", B_OpInf)


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
m = (
    model.m.detach()
    .numpy()
    .reshape(
        -1,
    )
)


# Quad-model
def model_quad_OpInf(t, x):
    """Define vector field of a quadratic system."""
    return A_OpInf @ (x - m) + H_OpInf @ np.kron(x - m, x - m) + B_OpInf


t_testing = np.linspace(0, 50, 500000)

if args.model_hypothesis == "globalstability":
    COLORS = [colors[0], colors[1]]
    TITLE = ["ground-truth", r"\texttt{atrMI}"]

else:
    COLORS = [colors[0], colors[2]]
    TITLE = ["ground-truth", r"\texttt{RK4-SINDy}"]

# Test initial conditions
INITS_CONDS = [[10.0, 10.0, -10.0], [100.0, -100.0, 100.0], [-500.0, 500.0, 500.0]]

for k, x0 in enumerate(INITS_CONDS):
    # we remark that the learning model is for shifting data, whereas original model is without. Therefore, learned model will take shifted data as initial condition.

    x0_no_scale = x0[:]

    x0[-1] = x0[-1] - 25.0
    for i in range(len(x0)):
        x0[i] = x0[i] / 8

    # Solving original model
    sol = solve_ivp(
        fun=lambda t, Y: lorenz_system(t, Y, sigma, rho, beta),
        t_span=[t_testing[0], t_testing[-1]],
        y0=x0_no_scale,
        t_eval=t_testing,
    )

    # Solving inferred model
    sol_OpInf = solve_ivp(
        model_quad_OpInf, [t_testing[0], t_testing[-1]], x0, t_eval=t_testing
    )

    full_sol_OpInf = sol_OpInf.y

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sol.y[0][::2], sol.y[1][::2], (sol.y[2][::2]), color=COLORS[0])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_title(TITLE[0])

    plt.tight_layout()

    fig.savefig(Params.path + f"simulation_test_orig{k}.pdf", bbox_inches="tight")
    fig.savefig(
        Params.path + f"simulation_test_orig{k}.png", dpi=300, bbox_inches="tight"
    )

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        sol_OpInf.y[0][::2] * 8,
        sol_OpInf.y[1][::2] * 8,
        (sol_OpInf.y[2][::2] * 8 + 25),
        color=COLORS[1],
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_title(TITLE[1])

    plt.tight_layout()

    fig.savefig(Params.path + f"simulation_test_learnt{k}.pdf", bbox_inches="tight")
    fig.savefig(
        Params.path + f"simulation_test_learnt{k}.png", dpi=300, bbox_inches="tight"
    )
