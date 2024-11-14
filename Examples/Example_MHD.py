import argparse
import os
from dataclasses import dataclass

import matplotlib
import matplotlib.gridspec as gridspec
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

reprod_seed(75)


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
        "./Results/MHD/NoStability/",
        module_models.ModelHypothesisNoStable_MHD,
    ),
    "globalstability": (
        "./Results/MHD/GlobalStability/",
        module_models.ModelHypothesisGlobalStable_MHD,
    ),
}

Params.path, model_hypothesis_function = path_funcs[args.model_hypothesis]

print(Params)
if not os.path.exists(Params.path):
    os.makedirs(Params.path)


# The following def is taken from [https://github.com/dynamicslab/pysindy/blob/master/examples/8_trapping_sindy_paper_examples.ipynb]
def make_lissajou(r, x_train, x_test, x_train_pred, x_test_pred, filename, colors):
    """Plot training and predicted data in a grid."""
    fig = plt.figure(figsize=(8, 8))
    spec = gridspec.GridSpec(ncols=r, nrows=r, figure=fig, hspace=0.0, wspace=0.0)
    XB = [r"$v_1$", r"$v_2$", r"$v_3$", r"$b_1$", r"$b_2$", r"$b_3$"]

    for i in range(r):
        for j in range(i, r):
            plt.subplot(spec[i, j])
            plt.plot(x_train[:, i], x_train[:, j], color=colors[0], linewidth=1)
            plt.plot(
                x_train_pred[:, i],
                x_train_pred[:, j],
                "--",
                color=colors[2],
                linewidth=1,
            )
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                plt.ylabel(r"$x_" + str(i) + r"$", fontsize=18)
                plt.ylabel(XB[i], fontsize=18)
            if i == r - 1:
                plt.xlabel(r"$x_" + str(j) + r"$", fontsize=18)
                plt.xlabel(XB[j], fontsize=18)
        for j in range(i):
            plt.subplot(spec[i, j])
            plt.plot(x_test[:, j], x_test[:, i], color=colors[1], linewidth=1)
            plt.plot(
                x_test_pred[:, j], x_test_pred[:, i], "--", color=colors[2], linewidth=1
            )
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                plt.ylabel(r"$y_" + str(i) + r"$", fontsize=18)
                plt.ylabel(XB[i], fontsize=18)
            if i == r - 1:
                plt.xlabel(r"$y_" + str(j) + r"$", fontsize=18)
                plt.xlabel(XB[j], fontsize=18)
    return fig


reprod_seed(75)


def mhd(t, x, nu=0.0, mu=0.0, sigma=0.0):
    """Define vector field of Carbone and Veltri triadic MHD model."""
    return [
        -2 * nu * x[0] + 4.0 * (x[1] * x[2] - x[4] * x[5]),
        -5 * nu * x[1] - 7.0 * (x[0] * x[2] - x[3] * x[5]),
        -9 * nu * x[2] + 3.0 * (x[0] * x[1] - x[3] * x[4]),
        -2 * mu * x[4] + 2.0 * (x[5] * x[1] - x[2] * x[4]),
        -5 * mu * x[4] + sigma * x[5] + 5.0 * (x[2] * x[3] - x[0] * x[5]),
        -9 * mu * x[5] + sigma * x[4] + 9.0 * (x[4] * x[0] - x[1] * x[3]),
    ]


t_train = np.linspace(0, 50, 5000)  # time-span

x0 = np.random.rand(6) - 0.5  # random-initial condition
x0_train = x0

for _ in range(
    25
):  # we tried to create an initial condition which provid more dynamics. Thats why there are 25 iterations
    x0_test = np.random.rand(6) - 0.5

# generating initial conditions
sol = solve_ivp(
    fun=lambda t, x: mhd(t, x), t_span=[t_train[0], t_train[-1]], y0=x0, t_eval=t_train
)

x_train = sol.y.T


######################################################################
# Preparing data for learning and dataloaders
X_training = sol.y.reshape(1, 6, -1)
t = t_train
X_training_true = X_training
print(X_training.shape)
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

temp_X = X[0]

for i in range(1, X.shape[0]):
    temp_X = np.hstack((temp_X, X[i]))


r = 6
temp_Xr = temp_X  # reduced data
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

######################################################################
# Defining model, optimizer, and training
Params.regularization_H = 1e-12  # regularizer for H
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

# Hard-pruning
TOL = 1e-1
for _ in range(4):
    print("\n")
    # Removing the coefficients smaller than tol and set gradients w.r.t. them to zero
    # so that they will not be updated in the iterations
    Ws = model._J.detach().clone()
    Mask_Ws = (Ws.abs() > TOL).type(torch.float)
    model._J = torch.nn.Parameter(Ws * Mask_Ws)
    model._J.register_hook(lambda grad: grad.mul_(Mask_Ws))

    Wh = model._H_tensor.detach().clone()
    Mask_Wh = (Wh.abs() > 10 * TOL).type(torch.float)
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


################################################################################

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


def model_quad_OpInf(t, x):
    """Define vector field of a quadratic system."""
    return A_OpInf @ (x - m) + H_OpInf @ np.kron(x - m, x - m) + B_OpInf


t_testing = np.linspace(
    0, 50, 5000
)  # training time-span which is differntial than training
# re-defining the plot-settings (somehow it works better to re-load again!)


if args.model_hypothesis == "globalstability":
    COLORS_ = colors[8]
    TITLE = [r"\texttt{atrMI}"]

else:
    COLORS_ = "k"
    TITLE = [r"\texttt{RK4-SINDy}"]


INITS_CONDS = [x0_test]  # test initial condition

for _, x0 in enumerate(INITS_CONDS):
    # ground-truth model
    sol = solve_ivp(
        fun=lambda t, x: mhd(t, x),
        t_span=[t_testing[0], t_testing[-1]],
        y0=x0[:],
        t_eval=t_testing,
    )

    # learned model
    sol_OpInf = solve_ivp(
        model_quad_OpInf, [t_testing[0], t_testing[-1]], x0, t_eval=t_testing
    )
    full_sol_OpInf = sol_OpInf.y

sol_train = solve_ivp(
    model_quad_OpInf, [t_testing[0], t_testing[-1]], x0_train, t_eval=t_testing
)
full_sol_train = sol_train.y

# Making ploting
fig = make_lissajou(
    6,
    x_train,
    sol.y.T,
    sol_train.y.T,
    sol_OpInf.y.T,
    "mhd",
    colors=[colors[0], "r", COLORS_],
)

fig.savefig(Params.path + "simulation_test.pdf", bbox_inches="tight")
fig.savefig(Params.path + "simulation_test.png", dpi=300, bbox_inches="tight")
