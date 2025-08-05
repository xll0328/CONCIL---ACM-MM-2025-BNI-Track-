

import torch
from torch.nn import functional as F
from typing import Optional, Union
from abc import abstractmethod, ABCMeta


class AnalyticLinear(torch.nn.Linear, metaclass=ABCMeta):
    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super(torch.nn.Linear, self).__init__()  # Skip the Linear class
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gamma: float = gamma
        self.bias: bool = bias
        self.dtype = dtype

        # Linear Layer
        if bias:
            in_features += 1
        weight = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

    @torch.inference_mode()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight)
        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1).to(X)), dim=-1)
        return X @ self.weight

    @property
    def in_features(self) -> int:
        if self.bias:
            return self.weight.shape[0] - 1
        return self.weight.shape[0]

    @property
    def out_features(self) -> int:
        return self.weight.shape[1]

    def reset_parameters(self) -> None:
        # Following the equation (4) of ACIL, self.weight is set to \hat{W}_{FCN}^{-1}
        self.weight = torch.zeros((self.weight.shape[0], 0)).to(self.weight)

    @abstractmethod
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        raise NotImplementedError()

    def update(self) -> None:
        assert torch.isfinite(self.weight).all(), (
            "Pay attention to the numerical stability! "
            "A possible solution is to increase the value of gamma. "
            "Setting self.dtype=torch.double also helps."
        )


class RecursiveLinear(AnalyticLinear):
    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super().__init__(in_features, gamma, bias, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        # Regularized Feature Autocorrelation Matrix (RFAuM)
        self.R: torch.Tensor
        R = torch.eye(self.weight.shape[0], **factory_kwargs) / self.gamma
        self.register_buffer("R", R)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """The core code of the ACIL and the G-ACIL.
        This implementation, which is different but equivalent to the equations shown in [1],
        is proposed in the G-ACIL [4], which supports mini-batch learning and the general CIL setting.
        """
        X, Y = X.to(self.weight), Y.to(self.weight)
        if self.bias:
            X = torch.cat((X, torch.ones(X.shape[0], 1).to(X)), dim=-1)

        num_targets = Y.shape[1]
        if num_targets > self.out_features:
            increment_size = num_targets - self.out_features
            tail = torch.zeros((self.weight.shape[0], increment_size)).to(self.weight)
            self.weight = torch.cat((self.weight, tail), dim=1)
        elif num_targets < self.out_features:
            increment_size = self.out_features - num_targets
            tail = torch.zeros((Y.shape[0], increment_size)).to(Y)
            Y = torch.cat((Y, tail), dim=1)

        # Please update your PyTorch & CUDA if the `cusolver error` occurs.
        # If you insist on using this version, doing the `torch.inverse` on CPUs might help.
        # K_inv = torch.eye(X.shape[0]).to(X) + X @ self.R @ X.T
        # K = torch.inverse(K_inv.cpu()).to(self.weight.device)
        K = torch.inverse(torch.eye(X.shape[0]).to(X) + X @ self.R @ X.T)
        # Equation (10) of ACIL
        self.R -= self.R @ X.T @ K @ X @ self.R
        # Equation (9) of ACIL
        # import pdb
        # pdb.set_trace()  # 设置断点  
        self.weight += self.R @ X.T @ (Y - X @ self.weight)


class GeneralizedARM(AnalyticLinear):
    """Analytic Re-weighting Module (ARM) for generalized CIL."""

    def __init__(
        self,
        in_features: int,
        gamma: float = 1e-1,
        bias: bool = False,
        device: Optional[Union[torch.device, str, int]] = None,
        dtype=torch.double,
    ) -> None:
        super().__init__(in_features, gamma, bias, device, dtype)
        factory_kwargs = {"device": device, "dtype": dtype}

        weight = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("weight", weight)

        A = torch.zeros((0, in_features, in_features), **factory_kwargs)
        self.register_buffer("A", A)

        C = torch.zeros((in_features, 0), **factory_kwargs)
        self.register_buffer("C", C)

        self.cnt = torch.zeros(0, dtype=torch.int, device=device)

    @property
    def out_features(self) -> int:
        return self.C.shape[1]

    @torch.inference_mode()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        X = X.to(self.weight)
        # Bias
        if self.bias:
            X = torch.concat((X, torch.ones(X.shape[0], 1)), dim=-1)

        # GCIL
        num_targets = int(y.max()) + 1
        if num_targets > self.out_features:
            increment_size = num_targets - self.out_features
            torch.cuda.empty_cache()
            # Increment C
            tail = torch.zeros((self.C.shape[0], increment_size)).to(self.weight)
            self.C = torch.concat((self.C, tail), dim=1)
            # Increment cnt
            tail = torch.zeros((increment_size,)).to(self.cnt)
            self.cnt = torch.concat((self.cnt, tail))
            # Increment A
            tail = torch.zeros((increment_size, self.in_features, self.in_features))
            self.A = torch.concat((self.A, tail.to(self.A)))
            torch.cuda.empty_cache()
        else:
            num_targets = self.out_features

        # ACIL
        Y = F.one_hot(y, max(num_targets, num_targets)).to(self.C)
        self.C += X.T @ Y

        # Label Balancing
        y_labels, label_cnt = torch.unique(y, sorted=True, return_counts=True)
        y_labels, label_cnt = y_labels.to(self.cnt.device), label_cnt.to(
            self.cnt.device
        )
        self.cnt[y_labels] += label_cnt

        # Accumulate
        for i in range(num_targets):
            X_i = X[y == i]
            self.A[i] += X_i.T @ X_i

    @torch.inference_mode()
    def update(self):
        cnt_inv = 1 / self.cnt.to(self.dtype)
        cnt_inv[torch.isinf(cnt_inv)] = 0  # replace inf with 0
        cnt_inv *= len(self.cnt) / cnt_inv.sum()

        weighted_A = torch.sum(cnt_inv[:, None, None].mul(self.A), dim=0)
        A = weighted_A + self.gamma * torch.eye(self.in_features).to(self.A)
        C = self.C.mul(cnt_inv[None, :])

        self.weight = torch.inverse(A) @ C
