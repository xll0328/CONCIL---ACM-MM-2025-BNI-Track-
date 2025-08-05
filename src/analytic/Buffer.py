
import torch
from typing import Optional, Union, Callable
from abc import ABCMeta, abstractmethod


activation_t = Union[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module]


class Buffer(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class RandomBuffer(torch.nn.Linear, Buffer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=torch.float,
        activation: Optional[activation_t] = torch.relu_,
    ) -> None:
        super(torch.nn.Linear, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.activation: activation_t = (
            torch.nn.Identity() if activation is None else activation
        )

        W = torch.empty((out_features, in_features), **factory_kwargs)
        b = torch.empty(out_features, **factory_kwargs) if bias else None
        
        # 初始化时创建一个随机权重矩阵 W 和偏置向量 b，并将其注册为缓冲区（register_buffer）。
        # Using buffer instead of parameter
        self.register_buffer("weight", W)
        self.register_buffer("bias", b)

        # Random Initialization
        self.reset_parameters()

    # forward 方法对输入 X 进行线性变换，并应用激活函数（默认为 ReLU）。
    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.weight)

        # 动态调整 weight 的形状
        if X.shape[1] != self.weight.shape[1]:
            new_weight = torch.empty((self.out_features, X.shape[1]), device=self.weight.device, dtype=self.weight.dtype)
            # # 只初始化新增的权重
            # if X.shape[1] > self.weight.shape[1]:
            #     new_weight[:, :self.weight.shape[1]] = self.weight
            #     new_weight[:, self.weight.shape[1]:] = torch.randn((self.out_features, X.shape[1] - self.weight.shape[1]), device=self.weight.device, dtype=self.weight.dtype)
            # else:
            #     new_weight = self.weight[:, :X.shape[1]]


            # 初始化所有的权重
            if X.shape[1] > self.weight.shape[1]:
                new_weight[:, :self.weight.shape[1]] = self.weight
                new_weight[:, self.weight.shape[1]:] = torch.randn((self.out_features, X.shape[1] - self.weight.shape[1]), device=self.weight.device, dtype=self.weight.dtype)
            else:
                new_weight[:, :self.weight.shape[1]] = self.weight
                new_weight[:, self.weight.shape[1]:] = torch.randn((self.out_features, X.shape[1] - self.weight.shape[1]), device=self.weight.device, dtype=self.weight.dtype)

            self.weight = new_weight
            
        return self.activation(super().forward(X))


class GaussianKernel(Buffer):
    def __init__(
        self, mean: torch.Tensor, sigma: float = 1, device=None, dtype=torch.float
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": dtype}
        assert len(mean.shape) == 2, "The mean should be a 2D tensor."
        mean = mean[None, :, :].to(**factory_kwargs)
        beta = 1 / (2 * (sigma**2))
        self.register_buffer("mean", mean)
        self.register_buffer("beta", torch.tensor(beta, **factory_kwargs))

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.square_(torch.cdist(X.to(self.mean), self.mean))
        return torch.exp_(X.mul_(-self.beta))

    def init(self, X: torch.Tensor, size: Optional[int] = None) -> None:
        if size is not None:
            if size <= X.shape[0]:
                idx = torch.randperm(size).to(X.device)
                X = X[idx]
            else:
                # The buffer size is suggested to be greater than the number of initial samples.
                # Generate center vectors randomly
                n_require = size - X.shape[0]
                W_proj = torch.normal(mean=0, std=1, size=(n_require, X.shape[0])).to(X)
                W_proj /= torch.sum(W_proj, dim=0)
                X = torch.cat([X, W_proj @ X], dim=0)
        self.mean = X.to(self.mean)
