import abc
import torch
from torch import nn
from torch.nn import functional as F
from utils import get_data_loader
import copy
import numpy as np
import tqdm


class AdaptiveProjection(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        # 1. 调用父类的 __init__
        super().__init__()

        # 2. 设置你自己的参数
        self.lambda_0 = 0.5
        self.gamma = 1.0
        self.M = 10
        self.epsilon_0 = 1e-6
        self.scaling_power = 0.33

        # 3. 初始化状态变量
        self.prev_theta_p = None
        self.current_H_p = None
        self.prev_H_p = None
        self.lambda_t = self.lambda_0

        # 4. 定义需要正则化的参数
        self.regularized_param_names = [
            n for n, p in self.named_parameters()
            if p.requires_grad and 'bn' not in n and 'bias' not in n
        ]
        print("AdaptiveProjection: 将对以下参数进行正则化:", self.regularized_param_names)

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    ####----adaptive peojection----####


    def _update_lambda_t(self):
        """在每个训练步中更新 lambda_t。"""
        if self.prev_theta_p is None or self.current_H_p is None or self.prev_H_p is None:
            return

        try:
            total_hessian_diff_norm_sq = 0.0
            total_param_diff_norm_sq = 0.0
            num_params_total = 0.0

            for name in self.regularized_param_names:
                if name not in self.current_H_p or name not in self.prev_H_p or name not in self.prev_theta_p: continue

                h_diff = self.current_H_p[name] - self.prev_H_p[name]
                total_hessian_diff_norm_sq += torch.norm(h_diff, p='fro') ** 2

                p_diff = self.model.get_parameter(name).data - self.prev_theta_p[name]
                total_param_diff_norm_sq += torch.norm(p_diff) ** 2
                num_params_total += self.model.get_parameter(name).numel()

            if num_params_total > 0:
                numerator = torch.sqrt(total_hessian_diff_norm_sq) / num_params_total
                denominator = self.gamma * (torch.sqrt(total_param_diff_norm_sq) / num_params_total)
                denominator = torch.clamp(denominator, min=1e-8)
                ratio = numerator / denominator

                effective_ratio = torch.pow(torch.clamp(ratio, min=1e-9), self.scaling_power)
                lambda_hat = (self.lambda_0 * effective_ratio).clamp(min=self.epsilon_0, max=self.M)
                self.lambda_t = lambda_hat.item()
        except Exception as e:
            print(f"Error in λ_t update: {e}")

    def penalty(self):
        """计算正则化惩罚项。这个方法会被父类的 train_a_batch 自动调用。"""
        # 只有在非首任务且 lambda_t > 0 时才计算惩罚
        if self.prev_theta_p is None or self.lambda_t == 0:
            return torch.tensor(0.0).to(self._device())

        # 在计算惩罚前，先更新 lambda_t
        self._update_lambda_t()

        penalty = 0.0
        for name, param in self.model.named_parameters():
            if name in self.regularized_param_names and name in self.prev_theta_p:
                p_diff_sq = (param - self.prev_theta_p[name]).pow(2)
                penalty += p_diff_sq.sum()

        return 0.5 * self.lambda_t * penalty

    def estimate_fisher(self, dataset, allowed_classes=None):
        """利用 train_cl 提供的任务结束钩子来更新我们的状态。"""
        # 只有在需要正则化时才执行
        if self.lambda_0 > 0:
            print(f"\n>>> AdaptiveProjection: 任务 {self.model.context_id} 结束, 更新状态...")

            # 1. 将上一个任务的 H_current 存为 prev_H_p
            if self.current_H_p is not None:
                self.prev_H_p = copy.deepcopy(self.current_H_p)

            # 2. 计算当前刚结束任务的 Hessian，作为下一个任务的 current_H_p
            #    注意：这里的 `dataset` 是 train_cl 传入的当前任务的数据集
            self.current_H_p = self.compute_hessian(dataset, allowed_classes)

            # 3. 保存当前训练完的参数，作为下一个任务的 prev_theta_p
            self.prev_theta_p = {
                n: p.clone().detach() for n, p in self.model.named_parameters() if n in self.regularized_param_names
            }
            print(">>> AdaptiveProjection: 状态更新完毕。")

    def compute_hessian(self, dataset, allowed_classes=None):
        # ... (这个函数保持原样，但确保 create_graph=False, retain_graph=False)
        # ...
        # grad = torch.autograd.grad(loss, param, retain_graph=False, create_graph=False)[0]
        # ...
        # (请确保这里的修改，因为我们不再需要构建计算图)
        model = self.model
        model.eval()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch, shuffle=True)
        hessian = {}
        for name, param in model.named_parameters():
            if name not in self.regularized_param_names: continue

            grad_sum = torch.zeros_like(param.data)
            if len(param.shape) > 1:
                if grad_sum.dim() != param.dim() - 1:  # 修正一个可能的bug
                    grad_sum = torch.zeros_like(param.data.sum(dim=0))

            total_samples = 0
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)

                if allowed_classes is not None:
                    indices = [i for i, label in enumerate(y) if label.item() in allowed_classes]
                    if not indices: continue
                    x, y = x[indices], y[indices]

                model.zero_grad()
                output = model(x, allowed_classes=allowed_classes)
                loss = self.criterion(output, y)

                grad = torch.autograd.grad(loss, param, retain_graph=False, create_graph=False)[0]

                if len(param.shape) > 1:
                    grad_sum += grad.detach().pow(2).sum(dim=0)
                else:
                    grad_sum += grad.detach().pow(2)
                total_samples += x.size(0)

            if total_samples > 0:
                hessian[name] = grad_sum / total_samples
            else:
                hessian[name] = torch.zeros_like(grad_sum) + 1e-8
                print(f"Warning: No samples for Hessian computation of {name}")

        model.train()
        return hessian
