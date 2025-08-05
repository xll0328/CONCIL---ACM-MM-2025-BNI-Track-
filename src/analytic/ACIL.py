# -*- coding: utf-8 -*-
"""
Implementation of the ACIL [1] and the G-ACIL [2].
The G-ACIL is a generalization of the ACIL in the generalized setting.
For the popular setting, the G-ACIL is equivalent to the ACIL.

References:
[1] Zhuang, Huiping, et al.
    "ACIL: Analytic class-incremental learning with absolute memorization and privacy protection."
    Advances in Neural Information Processing Systems 35 (2022): 11602-11614.
[2] Zhuang, Huiping, et al.
    "G-ACIL: Analytic Learning for Exemplar-Free Generalized Class Incremental Learning"
    arXiv preprint arXiv:2403.15706 (2024).
"""

import torch
from os import path
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence
from .utils import set_weight_decay, validate
from torch._prims_common import DeviceLikeType
from .Buffer import RandomBuffer
from torch.nn import DataParallel
from .Learner import Learner, loader_t
from .AnalyticLinear import AnalyticLinear, RecursiveLinear


class ACIL(torch.nn.Module):
# backbone_output: 特征提取器的输出维度。
# backbone: 特征提取器，默认为 torch.nn.Flatten()，将输入展平为一维向量。
# buffer_size: 缓冲区的大小，默认为 8192。
# gamma: 正则化参数，默认为 1e-3。
# device: 设备类型，默认为 None，表示使用默认设备。
# dtype: 数据类型，默认为 torch.double。
# linear: 线性分析层的类型，默认为 RecursiveLinear。
    def __init__(
        self,
        backbone_output: int,
        backbone: torch.nn.Module = torch.nn.Flatten(),
        buffer_size: int = 8192,
        gamma: float = 1e-3,
        device=None,
        dtype=torch.double,
        linear: type[AnalyticLinear] = RecursiveLinear,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        # 初始化 backbone 特征提取器和 backbone_output 输出维度。
        self.backbone = backbone
        self.backbone_output = backbone_output

        # 初始化 buffer，使用 RandomBuffer 类，将特征提取器的输出存储在缓冲区中。
        self.buffer_size = buffer_size
        self.buffer = RandomBuffer(backbone_output, buffer_size, **factory_kwargs)
        
        # 初始化 analytic_linear，使用指定的线性分析层类型（默认为 RecursiveLinear）
        self.analytic_linear = linear(buffer_size, gamma, **factory_kwargs)
        
        # 后续无训练过程了，全是求参数矩阵的解析解
        self.eval()

    ## 对输入 X 进行特征提取（self.backbone(X)），然后将提取的特征传递给缓冲区 self.buffer 进行扩展。
    @torch.no_grad()
    def feature_expansion(self, X: torch.Tensor) -> torch.Tensor:
        return self.buffer(self.backbone(X))

    ## 特征传递给线性分析层 
    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.analytic_linear(self.feature_expansion(X))


    ## 拟合求解
    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> None:
        # 将标签 y 转换为 one-hot 编码形式 Y
        Y = torch.nn.functional.one_hot(y)
        # 对输入 X 进行特征扩展
        X = self.feature_expansion(X)
        # 调用线性分析层的 fit 方法，将扩展后的特征 X 和 one-hot 编码的标签 Y 传递给线性分析层进行拟合。
        self.analytic_linear.fit(X, Y)





    # 调用线性分析层的 update 方法，进行内部参数的更新。
    @torch.no_grad()
    def update(self) -> None:
        self.analytic_linear.update()


class ACILLearner(Learner):
    """
    This implementation is for the G-ACIL [2], a general version of the ACIL [1] that
    supports mini-batch learning and the general CIL setting.
    In the traditional CIL settings, the G-ACIL is equivalent to the ACIL.
    """

    def __init__(
        self,
        args: Dict[str, Any],
        backbone: torch.nn.Module,
        backbone_output: int,
        device=None,
        all_devices: Optional[Sequence[DeviceLikeType]] = None,
    ) -> None:
        super().__init__(args, backbone, backbone_output, device, all_devices)
        self.learning_rate: float = args["learning_rate"]
        self.buffer_size: int = args["buffer_size"]
        self.gamma: float = args["gamma"]
        self.base_epochs: int = args["base_epochs"]
        self.warmup_epochs: int = args["warmup_epochs"]
        self.make_model()

    def base_training(
        self,
        train_loader: loader_t,
        val_loader: loader_t,
        baseset_size: int,
    ) -> None:
        # 创建一个包含特征提取器和线性层的模型，并将其移动到指定设备。
        model = torch.nn.Sequential(
            self.backbone,
            torch.nn.Linear(self.backbone_output, baseset_size),
        ).to(self.device, non_blocking=True)

        # ？？并行的，没必要吧，我就只有单卡
        model = self.wrap_data_parallel(model)

        # 使用 SGD 优化器和余弦退火学习率调度器进行训练
        if self.args["separate_decay"]:
            params = set_weight_decay(model, self.args["weight_decay"])
        else:
            params = model.parameters()
        optimizer = torch.optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=self.args["momentum"],
            weight_decay=self.args["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.base_epochs - self.warmup_epochs, eta_min=1e-6 # type: ignore
        )
        if self.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                total_iters=self.warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_scheduler, scheduler], [self.warmup_epochs]
            )

        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=self.args["label_smoothing"]
        ).to(self.device, non_blocking=True)

        best_acc = 0.0
        logging_file_path = path.join(self.args["saving_root"], "base_training.csv")
        logging_file = open(logging_file_path, "w", buffering=1)
        print(
            "epoch",
            "best_acc@1",
            "loss",
            "acc@1",
            "acc@5",
            "f1-micro",
            "training_loss",
            "training_acc@1",
            "training_acc@5",
            "training_f1-micro",
            "training_learning-rate",
            file=logging_file,
            sep=",",
        )

        for epoch in range(self.base_epochs + 1):
            if epoch != 0:
                print(
                    f"Base Training - Epoch {epoch}/{self.base_epochs}",
                    f"(Learning Rate: {optimizer.state_dict()['param_groups'][0]['lr']})",
                )
                model.train()
                for X, y in tqdm(train_loader, "Training"):
                    X: torch.Tensor = X.to(self.device, non_blocking=True)
                    y: torch.Tensor = y.to(self.device, non_blocking=True)
                    assert y.max() < baseset_size

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(X)
                    loss: torch.Tensor = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

            # Validation on training set
            model.eval()
            train_meter = validate(
                model, train_loader, baseset_size, desc="Training (Validation)"
            )
            print(
                f"loss: {train_meter.loss:.4f}",
                f"acc@1: {train_meter.accuracy * 100:.3f}%",
                f"acc@5: {train_meter.accuracy5 * 100:.3f}%",
                f"f1-micro: {train_meter.f1_micro * 100:.3f}%",
                sep="    ",
            )

            val_meter = validate(model, val_loader, baseset_size, desc="Testing")
            if val_meter.accuracy > best_acc:
                best_acc = val_meter.accuracy
                if epoch != 0:
                    self.save_object(
                        (self.backbone, X.shape[1], self.backbone_output),
                        "backbone.pth",
                    )

            # Validation on testing set
            print(
                f"loss: {val_meter.loss:.4f}",
                f"acc@1: {val_meter.accuracy * 100:.3f}%",
                f"acc@5: {val_meter.accuracy5 * 100:.3f}%",
                f"f1-micro: {val_meter.f1_micro * 100:.3f}%",
                f"best_acc@1: {best_acc * 100:.3f}%",
                sep="    ",
            )
            print(
                epoch,
                best_acc,
                val_meter.loss,
                val_meter.accuracy,
                val_meter.accuracy5,
                val_meter.f1_micro,
                train_meter.loss,
                train_meter.accuracy,
                train_meter.accuracy5,
                train_meter.f1_micro,
                optimizer.state_dict()["param_groups"][0]["lr"],
                file=logging_file,
                sep=",",
            )
        logging_file.close()
        # 保存最佳模型并重新创建 ACIL 模型。
        self.backbone.eval()
        self.make_model()

    def make_model(self) -> None:
        self.model = ACIL(
            self.backbone_output,
            self.wrap_data_parallel(self.backbone),
            self.buffer_size,
            self.gamma,
            device=self.device,
            dtype=torch.double,
            linear=RecursiveLinear,
        )

    # 进行增量学习，逐批次处理数据并调用 model.fit 方法进行拟合
    @torch.no_grad()
    def learn(
        self,
        data_loader: loader_t,
        incremental_size: int,
        desc: str = "Incremental Learning",
    ) -> None:
        self.model.eval()
        for X, y in tqdm(data_loader, desc=desc):
            X: torch.Tensor = X.to(self.device, non_blocking=True)
            y: torch.Tensor = y.to(self.device, non_blocking=True)
            self.model.fit(X, y, increase_size=incremental_size)

    def before_validation(self) -> None:
        self.model.update()

    def inference(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    @torch.no_grad()
    def wrap_data_parallel(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.all_devices is not None and len(self.all_devices) > 1:
            return DataParallel(model, self.all_devices, output_device=self.device) # type: ignore
        return model
