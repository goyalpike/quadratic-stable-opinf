#!/usr/bin/env python3
import sys

import numpy as np
import torch
import torch.nn as nn

from qs_opinf.utils import rk4th_onestep


def training(model, dataloaders, opt_func, Params, scheduler=None):
    """Given model, dataloaders and optimizer, it optimizes models/parameters.

    Args:
        model: Neural network model
        dataloaders: Training data
        opt_func: Optimizer
        Params: Parameters (e.g., number of epochs)
        scheduler (optional): Learning rate sheduler. Defaults to None.

    Returns
    -------
        (model, loss): Trained model and loss wrt epoch.
    """
    print("#" * 75)
    criteria = nn.MSELoss()
    loss_track = []
    for g in range(Params.num_epochs):
        for y in dataloaders["train"]:
            opt_func.zero_grad()
            total_loss = 0.0
            for i in range(y[0].shape[0]):
                yi = y[0][i]
                timesteps_i = torch.tensor(np.diff(y[1][i], axis=0)).float()
                y_total = yi

                ##################################
                # One forward step predictions
                ##################################
                y_pred = rk4th_onestep(model, y_total[:-1], timestep=timesteps_i)

                total_loss += (1 / timesteps_i.mean()) * criteria(y_pred, y_total[1:])

                ##################################

            total_loss = total_loss / y[0].shape[0]
            loss_track.append(total_loss.item())
            total_loss += Params.regularization_H * model.H.abs().mean()
            # total_loss += Params.regularization_H * model.A.abs().mean()
            total_loss.backward()
            opt_func.step()
            if scheduler:
                scheduler.step()

        if g and (g + 1) % 100 == 0:
            sys.stdout.write(
                "\r [Epoch %d/%d] [Training loss: %.2e] [Learning rate: %.2e]"
                % (
                    g + 1,
                    Params.num_epochs,
                    loss_track[g],
                    opt_func.param_groups[0]["lr"],
                )
            )

    return model, loss_track
