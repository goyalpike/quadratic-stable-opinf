import torch
import torch.nn as nn
from qs_opinf.utils import kron


class ModelHypothesis(nn.Module):
    def __init__(self, sys_order, B_term=True, *args, **kwargs):
        super().__init__()
        print("Warning: No Stability Gurantees!")
        FAC = 10
        self.B_term = B_term
        self.A = torch.nn.Parameter(torch.randn(sys_order, sys_order) / FAC)
        self.B = torch.nn.Parameter(torch.zeros(sys_order, 1) / FAC)
        self.H = torch.nn.Parameter(torch.zeros(sys_order, sys_order**2) / FAC)
        print("B_term:", self.B_term)

    def forward(self, x, t):
        x_A = x @ self.A.T
        x_H = kron(x, x) @ self.H.T
        if self.B_term:
            return x_A + x_H + self.B.T
        else:
            return x_A + x_H


class ModelHypothesisLocalStable(nn.Module):
    def __init__(self, sys_order, B_term=True, *arg, **kwargs):
        super().__init__()
        """
            Define model parameters.
            A = (J-R)Q,
            where J is skew-symmetric, R and Q are positive semi-definite matrics.
            This then guarantees stability of A.
        """
        print("Local Stability Gurantees!")
        FAC = 10
        self.B_term = B_term
        self._J = torch.nn.Parameter(torch.randn(sys_order, sys_order) / FAC)
        self._R = torch.nn.Parameter(torch.randn(sys_order, sys_order) / FAC)
        self._Q = torch.nn.Parameter(torch.randn(sys_order, sys_order) / FAC)

        self.B = torch.nn.Parameter(torch.zeros(sys_order, 1))
        self.H = torch.nn.Parameter(torch.zeros(sys_order, sys_order**2))
        print("B_term:", self.B_term)

    @property
    def A(self):
        J = self._J - self._J.T
        R = self._R @ self._R.T

        Q = self._Q @ self._Q.T
        _A = (J - R) @ Q

        self._A = _A
        return self._A

    def forward(self, x, t):
        x_A = x @ self.A.T
        x_H = kron(x, x) @ self.H.T
        if self.B_term:
            return x_A + x_H + self.B.T
        else:
            return x_A + x_H


class ModelHypothesisGlobalStable(nn.Module):
    def __init__(self, sys_order, B_term=True, *arg, **kwargs):
        super().__init__()
        """
            Define model parameters.
            A = (J-R)Q,
            where J is skew-symmetric, R and Q are positive semi-definite matrics.
            We have assumed Q to be identity; hence, it is removed from the optimzer.
            This then guarantees stability of A.
        """
        print("Global Stability Gurantees!")
        FAC = 10
        self.B_term = B_term
        self.sys_order = sys_order
        self._J = torch.nn.Parameter(torch.randn(sys_order, sys_order) / FAC)
        self._R = torch.nn.Parameter(torch.randn(sys_order, sys_order) / FAC)

        self.B = torch.nn.Parameter(torch.zeros(sys_order, 1))
        self._H_tensor = torch.nn.Parameter(
            torch.zeros(sys_order, sys_order, sys_order) / FAC
        )
        print("B_term:", self.B_term)

    @property
    def A(self):
        J = self._J - self._J.T
        R = self._R @ self._R.T
        _A = J - R

        self._A = _A
        return self._A

    @property
    def H(self):
        _H_tensor2 = self._H_tensor.permute(0, 2, 1)
        J_tensor = self._H_tensor - _H_tensor2
        self._H = J_tensor.permute(1, 0, 2).reshape(self.sys_order, self.sys_order**2)
        return self._H

    def forward(self, x, t):
        x_A = x @ self.A.T
        x_H = kron(x, x) @ self.H.T
        if self.B_term:
            return x_A + x_H + self.B.T
        else:
            return x_A + x_H


class SINDy_NoStable(nn.Module):
    def __init__(self, sys_order, B_term=True, *arg, **kwargs):
        super().__init__()
        """
        Define model parameters.
        A = (J-R)Q,
        where J is skew-symmetric, R and Q are positive semi-definite matrics.
        This then guarantees stability of A.
        """
        print("No Stability Gurantees!")
        fac = 10
        self.B_term = B_term
        self.sys_order = sys_order
        self._J = torch.nn.Parameter(torch.randn(sys_order, sys_order) / fac)
        self._R = torch.zeros(sys_order, sys_order)

        self.B = torch.nn.Parameter(torch.zeros(sys_order, 1))
        self.m = torch.zeros(1, sys_order)
        self._H_tensor = torch.nn.Parameter(
            torch.zeros(sys_order, sys_order, sys_order) / fac
        )

    @property
    def A(self):
        return self._J

    @property
    def H(self):
        self._H = self._H_tensor.reshape(self.sys_order, self.sys_order**2)
        return self._H

    def forward(self, x, t):
        x_A = x @ self.A.T
        x_H = kron(x, x) @ self.H.T
        if self.B_term:
            return x_A + x_H + self.B.T
        else:
            return x_A + x_H


class SINDy_Stable_Trapping(nn.Module):
    def __init__(self, sys_order, B_term=True, *arg, **kwargs):
        super().__init__()
        """
        Define model parameters.
        A = (J-R)Q,
        where J is skew-symmetric, R and Q are positive semi-definite matrics.
        This then guarantees stability of A.
        """
        print("Global Stability Gurantees!")
        fac = 10
        self.B_term = B_term
        self.sys_order = sys_order
        self._J = torch.nn.Parameter(torch.randn(sys_order, sys_order) / fac)
        self._R = torch.nn.Parameter(torch.randn(sys_order, sys_order) / fac)
        self._Q = torch.nn.Parameter(torch.randn(sys_order, sys_order) / fac)

        self.B = torch.nn.Parameter(torch.zeros(sys_order, 1))
        self.m = torch.nn.Parameter(torch.zeros(1, sys_order))
        self._H_tensor = torch.nn.Parameter(
            torch.zeros(sys_order, sys_order, sys_order) / fac
        )

    @property
    def A(self):
        J = self._J - self._J.T
        R = self._R @ self._R.T + 0.00 * torch.eye(self.sys_order)

        Q = self._Q @ self._Q.T
        _A = J - R

        self._A = _A
        # self._A = self._J + self._J
        return self._A

    @property
    def H(self):
        _H_tensor2 = self._H_tensor.permute(0, 2, 1)
        J_tensor = self._H_tensor - _H_tensor2
        self._H = J_tensor.permute(1, 0, 2).reshape(self.sys_order, self.sys_order**2)
        Q = self._Q @ self._Q.T
        # self._H = self._H_tensor.reshape(self.sys_order,self.sys_order**2)
        return self._H

    def forward(self, x, t):
        x = x - self.m
        x_A = x @ self.A.T
        x_H = kron(x, x) @ self.H.T
        if self.B_term:
            return x_A + x_H + self.B.T
        else:
            return x_A + x_H


class ModelHypothesisNoStable_MHD(nn.Module):
    def __init__(self, sys_order, B_term=True, *arg, **kwargs):
        super().__init__()
        """
        Define model parameters.
        A = (J-R)Q,
        where J is skew-symmetric, R and Q are positive semi-definite matrics.
        This then guarantees stability of A.
        """
        print("No Stability Gurantees!")
        fac = 10
        self.B_term = B_term
        self.sys_order = sys_order
        self._J = torch.nn.Parameter(torch.randn(sys_order, sys_order) / fac)
        self._R = torch.zeros(sys_order, sys_order)

        self.B = torch.nn.Parameter(torch.zeros(sys_order, 1))
        self.m = torch.zeros(1, sys_order)
        self._H_tensor = torch.nn.Parameter(
            torch.zeros(sys_order, sys_order, sys_order) / fac
        )

    @property
    def A(self):
        return self._J

    @property
    def H(self):
        self._H = self._H_tensor.reshape(self.sys_order, self.sys_order**2)
        return self._H

    def forward(self, x, t):
        x_A = x @ self.A.T
        x_H = kron(x, x) @ self.H.T
        if self.B_term:
            return x_A + x_H + self.B.T
        else:
            return x_A + x_H


class ModelHypothesisGlobalStable_MHD(nn.Module):
    def __init__(self, sys_order, B_term=True, *arg, **kwargs):
        super().__init__()
        """
        Define model parameters.
        A = (J-R)Q,
        where J is skew-symmetric, R and Q are positive semi-definite matrics.
        This then guarantees stability of A.
        """
        print("Global Stability Gurantees!")
        fac = 10
        self.B_term = B_term
        self.sys_order = sys_order
        self._J = torch.nn.Parameter(torch.randn(sys_order, sys_order) / fac)
        self._R = torch.nn.Parameter(torch.randn(sys_order, sys_order) / fac)
        self._Q = torch.nn.Parameter(torch.randn(sys_order, sys_order) / fac)

        # self.B = torch.nn.Parameter(torch.zeros(sys_order,1))
        self.B = torch.zeros(sys_order, 1)
        self.m = torch.nn.Parameter(torch.zeros(1, sys_order))
        self._H_tensor = torch.nn.Parameter(
            torch.zeros(sys_order, sys_order, sys_order) / fac
        )

    @property
    def A(self):
        J = self._J - self._J.T

        _A = J

        self._A = _A
        # self._A = self._J + self._J
        return self._A

    @property
    def H(self):
        _H_tensor2 = self._H_tensor.permute(0, 2, 1)
        J_tensor = self._H_tensor - _H_tensor2
        self._H = J_tensor.permute(1, 0, 2).reshape(self.sys_order, self.sys_order**2)
        Q = self._Q @ self._Q.T
        # self._H = self._H_tensor.reshape(self.sys_order,self.sys_order**2)
        return self._H

    def forward(self, x, t):
        x = x - self.m
        x_A = x @ self.A.T
        x_H = kron(x, x) @ self.H.T
        if self.B_term:
            return x_A + x_H + self.B.T
        else:
            return x_A + x_H
