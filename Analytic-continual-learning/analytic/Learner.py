import torch
from os import path
from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader
from torch._prims_common import DeviceLikeType
from typing import Union, Dict, Any, Optional, Sequence

loader_t = DataLoader[Union[torch.Tensor, torch.Tensor]]


class Learner(metaclass=ABCMeta):


    # base_training、learn、before_validation 和 inference 方法用于定义学习器的训练、增量学习、验证前处理和推理过程。
    # save_object 方法用于保存模型。
    # __call__ 方法允许将 Learner 实例作为函数调用，调用 inference 方法进行推理。
    def __init__(
        self,
        args: Dict[str, Any],
        backbone: torch.nn.Module,
        backbone_output: int,
        device=None,
        all_devices: Optional[Sequence[DeviceLikeType]] = None,
    ) -> None:
        self.args = args
        self.backbone = backbone
        self.backbone_output = backbone_output
        self.device = device
        self.all_devices = all_devices
        self.model: torch.nn.Module

    @abstractmethod
    def base_training(
        self,
        train_loader: loader_t,
        val_loader: loader_t,
        baseset_size: int,
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def learn(
        self,
        data_loader: loader_t,
        incremental_size: int,
        desc: str = "Incremental Learning"
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def before_validation() -> None:
        raise NotImplementedError()

    @abstractmethod
    def inference(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def save_object(self, model, file_name: str) -> None:
        torch.save(model, path.join(self.args["saving_root"], file_name))

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.inference(X)
