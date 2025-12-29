import abc
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from utils import get_data_loader
from models import fc
from models.utils.ncl import additive_nearest_kf
import copy
from utils import checkattr
import math



class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    '''Abstract module to add continual learning capabilities to a classifier (e.g., param regularization, replay).'''

    def __init__(self, gamma = 1.0, lambda_0 = 0.3):
        super().__init__()

        # List with the methods to create generators that return the parameters on which to apply param regularization
        self.param_list = [self.named_parameters]  #-> lists the parameters to regularize with SI or diagonal Fisher
                                                   #   (default is to apply it to all parameters of the network)
        #-> with OWM or KFAC Fisher, only parameters in [self.fcE] and [self.classifier] are regularized

        # Optimizer (and whether it needs to be reset)
        self.optimizer = None
        self.optim_type = "adam"
        #--> self.[optim_type]   <str> name of optimizer, relevant if optimizer should be reset for every context
        self.optim_list = []
        #--> self.[optim_list]   <list>, if optimizer should be reset after each context, provide list of required <dicts>

        # Scenario, singlehead & negative samples
        self.scenario = 'task'       # which scenario will the model be trained on
        self.classes_per_context = 2 # number of classes per context
        self.singlehead = False      # if Task-IL, does the model have a single-headed output layer?
        self.neg_samples = 'all'     # if Class-IL, which output units should be set to 'active'?

        # LwF / Replay
        self.replay_mode = "none"    # should replay be used, and if so what kind? (none|current|buffer|all|generative)
        self.replay_targets = "hard" # should distillation loss be used? (hard|soft)
        self.KD_temp = 2.            # temperature for distillation loss
        self.use_replay = "normal"   # how to use the replayed data? (normal|inequality|both)
                                     # -inequality = use gradient of replayed data as inequality constraint for gradient
                                     #               of the current data (as in A-GEM; Chaudry et al., 2019; ICLR)
        self.eps_agem = 0.           # parameter that improves numerical stability of AGEM (if set slighly above 0)
        self.lwf_weighting = False   # LwF has different weighting of the 'stability' and 'plasticity' terms than replay

        # XdG:
        self.mask_dict = None        # -> <dict> with context-specific masks for each hidden fully-connected layer
        self.excit_buffer_list = []  # -> <list> with excit-buffers for all hidden fully-connected layers

        # Parameter-regularization
        self.weight_penalty = False  #-> options for regularization strength
                                          #   - false:      no regularization strength
                                          #   - true:       use fixed reg-strength
        self.reg_strength = 1.0      #-> hyperparam: how strong to weigh the weight penalty ("regularisation strength")
        self.precondition = False
        self.alpha = 1e-10          #-> small constant to stabilize inversion of the Fisher Information Matrix
                                    #   (this is used as hyperparameter in OWM)
        self.importance_weighting = 'fisher'  #-> Options for estimation of parameter importance:
                                              #   - 'fisher':   Fisher Information matrix (e.g., as in EWC, NCL)
                                              #   - 'si':       ... diagonal, online importance estimation ...
                                              #   - 'owm':      ...
        self.experiment = None
        self.use_adaptive = False
        self.regularization_strength = 0.0
        self.fisher_kfac = False    #-> whether to use a block-diagonal KFAC approximation to the Fisher Information
                                    #   (alternative is a diagonal approximation)
        self.fisher_n = None        #-> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.fisher_labels = "all"  #-> what label(s) to use for any given sample when calculating the FI matrix?
                                    #   - 'all':    use all labels, weighted according to their predicted probabilities
                                    #   - 'sample': sample one label to use, using predicted probabilities for sampling
                                    #   - 'pred':   use the predicted label (i.e., the one with highest predicted prob)
                                    #   - 'true':   use the true label (NOTE: this is also called "empirical FI")
        self.fisher_batch = 1       #-> batch size for estimating FI-matrix (should be 1, for best results)
                                    #   (different from 1 only works if [fisher_labels]='pred' or 'true')
        self.context_count = 0      #-> counts 'contexts' (if a prior is used, this is counted as the first context)
        self.data_size = None       #-> inverse prior (can be set to # samples per context, or used as hyperparameter)
        self.epsilon = 0.1          #-> dampening parameter (SI): bounds 'omega' when squared parameter-change goes to 0
        self.offline = False        #-> use separate penalty term per context (as in original EWC paper)
        self.gamma = gamma         #-> decay-term for old contexts' contribution to cummulative FI (as in 'Online EWC')
        self.randomize_fisher = False

        # adaptive:
        self.lambda_0 = lambda_0  # 初始正则化强度
        self.M = 1e5 # 正则化强度的最大值
        self.epsilon_0 = 1e-6  # 数值稳定性的小常数
        self.scaling_power = 1.2  # 自适应正则化的缩放因子
        self.lambda_t = self.lambda_0  # 当前正则化强度（自适应时更新）
        self.prev_theta_p = None  # 之前任务的参数值
        self.current_H_p = None  # 当前任务的 Hessian
        self.prev_H_p = None  # 之前任务的 Hessian
        self.lambda_t_history = []  # track lambda evolution (useful for diagnostics / GMA plots)
        self.lambda_t_history_since_reference = None  # Task-2-only trace used for GMA reporting
        self.regularized_param_names = []  # 需要正则化的参数
        self.shared_param_names = []  # CIFAR100 Task-IL 场景下共享的参数
        self.lambda_ema = self.lambda_0  # EMA 平滑后的 λ_t
        self.ema_beta = 0.95 # EMA 平滑系数
        self.warmup_steps = 100  # warmup 的迭代步数
        self.update_step_count = 0  # 记录更新次数
        self.conflict_beta = 0.9  # 平滑几何冲突评分的系数
        self.geometry_conflict_ema = None
        self.drift_ema = None
        self.min_drift_scale = 1e-4
        self.nominal_drift = 5e-3
        self.geometry_gain = 2.5
        self.geometry_cap = 25.0
        self.drift_exponent = 1.0


        self.ablation_fixed_numerator = False
        self.fixed_numerator_value = 0.1
        self.ablation_fixed_denominator = False
        self.fixed_denominator_value = 10.0

        # Geometry mismatch attack state
        self.gma_attack = False
        self.gma_gamma = 0.0
        self.gma_tag = None
        self.gma_reference_params = None
        self.gma_reference_importance = None
        self.gma_adv_loss_history = []

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    # ----------------- adaptive regularization -----------------#
    def initialize_regularized_params(self):
        self.regularized_param_names = [n for n, p in self.named_parameters() if p.requires_grad and 'bn' not in n and 'bias' not in n]
        print(f"[DEBUG] Initialized regularized_param_names: {self.regularized_param_names}")
        self.shared_param_names = [n for n, p in self.named_parameters() if p.requires_grad and 'bn' not in n and 'bias' not in n and 'classifier' not in n]
        print(f"[DEBUG] shared regularized_param_names: {self.shared_param_names}")

    def _rms_from_tensor_dict(self, tensor_dict, names):
        total_sq = 0.0
        elem_count = 0
        for name in names:
            if name not in tensor_dict:
                continue
            tensor = tensor_dict[name]
            if tensor is None:
                continue
            total_sq += torch.norm(tensor, p='fro').pow(2).item()
            elem_count += tensor.numel()
        if elem_count == 0:
            return 0.0
        return math.sqrt(total_sq / elem_count)

    def update_lambda(self, x, y):
        if not self.use_adaptive or self.prev_theta_p is None or self.current_H_p is None or self.prev_H_p is None:
            return

        try:
            params_to_use_for_hessian = self.regularized_param_names
            if self.experiment == 'CIFAR100' and self.scenario == 'task':
                params_to_use_for_hessian = self.shared_param_names

            # ===== 计算分子：几何冲突强度 =====
            geometry_signal = 0.0
            if self.ablation_fixed_numerator:
                numerator = float(self.fixed_numerator_value)
            else:
                total_hessian_diff_norm_sq = 0.0
                elem_count = 0
                for name in params_to_use_for_hessian:
                    if name not in self.current_H_p or name not in self.prev_H_p:
                        continue
                    h_diff = self.current_H_p[name] - self.prev_H_p[name]
                    total_hessian_diff_norm_sq += torch.norm(h_diff, p='fro').pow(2).item()
                    elem_count += h_diff.numel()
                if elem_count == 0:
                    return
                numerator = math.sqrt(total_hessian_diff_norm_sq / elem_count)

                prev_scale = self._rms_from_tensor_dict(self.prev_H_p, params_to_use_for_hessian)
                curr_scale = self._rms_from_tensor_dict(self.current_H_p, params_to_use_for_hessian)
                reference_scale = max(0.5 * (prev_scale + curr_scale), self.epsilon_0)
                geometry_ratio = max(numerator / reference_scale, 0.0)
                geometry_signal = math.log1p(geometry_ratio)
                if self.geometry_conflict_ema is None:
                    self.geometry_conflict_ema = geometry_signal
                else:
                    self.geometry_conflict_ema = (
                            self.conflict_beta * self.geometry_conflict_ema +
                            (1 - self.conflict_beta) * geometry_signal
                    )

            # ===== 计算分母：参数漂移尺度 =====
            if self.ablation_fixed_denominator:
                drift_scale = float(self.fixed_denominator_value)
            else:
                total_param_diff_norm_sq = 0.0
                num_params_total = 0
                named_params_dict = dict(self.named_parameters())
                for name in params_to_use_for_hessian:
                    if name not in self.prev_theta_p or name not in named_params_dict:
                        continue
                    p_diff = named_params_dict[name] - self.prev_theta_p[name]
                    total_param_diff_norm_sq += p_diff.pow(2).sum().item()
                    num_params_total += named_params_dict[name].numel()
                if num_params_total == 0:
                    return
                drift_rms = math.sqrt(total_param_diff_norm_sq / num_params_total)
                if self.drift_ema is None:
                    self.drift_ema = drift_rms
                else:
                    self.drift_ema = (
                            self.conflict_beta * self.drift_ema +
                            (1 - self.conflict_beta) * drift_rms
                    )
                drift_scale = max(drift_rms, self.drift_ema if self.drift_ema is not None else 0.0,
                                  self.min_drift_scale)

            # ===== 计算 λ_t =====
            smoothed_geometry = self.geometry_conflict_ema if self.geometry_conflict_ema is not None else geometry_signal
            smoothed_geometry = max(smoothed_geometry, geometry_signal)
            scaled_geometry = min(max(smoothed_geometry, 0.0) * self.geometry_gain, self.geometry_cap)
            drift_normalized = (drift_scale / max(self.nominal_drift, self.epsilon_0)) ** self.drift_exponent
            stability_gate = 1.0 - math.exp(-drift_normalized)
            stability_gate = min(max(stability_gate, 0.0), 1.0)
            combined_score = scaled_geometry * stability_gate
            lambda_hat = self.lambda_0 * math.exp(self.scaling_power * combined_score)
            lambda_hat = max(min(lambda_hat, self.M), self.lambda_0)

            # Warmup + EMA
            self.update_step_count += 1
            warmup_factor = min(1.0, self.update_step_count / self.warmup_steps)
            lambda_hat = self.lambda_0 + warmup_factor * (lambda_hat - self.lambda_0)
            self.lambda_ema = self.ema_beta * self.lambda_ema + (1 - self.ema_beta) * lambda_hat
            self.lambda_t = self.lambda_ema
            self.lambda_t_history.append(float(self.lambda_t))
            if self.lambda_t_history_since_reference is not None:
                self.lambda_t_history_since_reference.append(float(self.lambda_t))


        except Exception as e:
            print(f"Error in λ_t update: {e}")

    def compute_hessian(self, dataset, batch_mode=False):
        """
        Compute a diagonal Hessian approximation for the dataset EFFICIENTLY
        while being MATHEMATICALLY EQUIVALENT to the original sample-by-sample version.
        """
        # 1. 设置模型为评估模式
        self.train(False)

        # 2. 使用一个合理的批次大小来加载数据
        #    这个批次仅用于数据传输，核心计算仍然是逐样本的
        hessian_batch_size = 256  # 可根据显存调整
        data_loader = get_data_loader(
            dataset, batch_size=hessian_batch_size, cuda=self._is_on_cuda()
        )

        # 3. 一次性为所有需要正则化的参数初始化累加器
        hessian = {
            name: torch.zeros_like(p.data)
            for name, p in self.named_parameters()
            if name in self.regularized_param_names
        }
        # 兼容您对多维参数的特殊处理 (sum over dim 0)
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in hessian and len(param.shape) > 1:
                    if hessian[name].dim() != param.dim() - 1:
                        hessian[name] = torch.zeros_like(param.data.sum(dim=0))

        # 4. 遍历数据加载器中的所有批次
        for x, y in data_loader:
            x, y = x.to(self._device()), y.to(self._device())

            # 5. 🚀 核心优化点：在批次内部进行“微循环”
            #    这保证了我们是逐样本计算梯度，与原始逻辑完全一致
            for i in range(x.size(0)):
                # 取出单个样本
                x_i, y_i = x[i].unsqueeze(0), y[i].unsqueeze(0)

                # 为这个【单个样本】进行一次完整的前向和后向传播
                self.zero_grad()
                output = self(x_i)
                loss = F.cross_entropy(output, y_i)
                loss.backward()

                # 6. 🚀 性能来源：一次性收集所有参数的梯度并累加其平方
                with torch.no_grad():
                    for name, param in self.named_parameters():
                        if name in hessian and param.grad is not None:
                            if len(param.shape) > 1 and hessian[name].dim() < param.grad.dim():
                                hessian[name] += param.grad.pow(2).sum(dim=0)
                            else:
                                hessian[name] += param.grad.pow(2)

        # 7. 最终，用数据集的总样本数进行归一化
        num_samples = len(dataset)
        if num_samples > 0:
            for name in hessian.keys():
                hessian[name] /= num_samples
        else:
            print("Warning: No samples for Hessian computation")
            # 按需处理
            for name in hessian.keys():
                hessian[name].fill_(1e-8)

        # 8. 恢复模型为训练模式
        self.train(True)
        torch.cuda.empty_cache()

        return hessian
    '''
    def update_adaptive_states(self, dataset=None):
        if not self.use_adaptive:
            return
        print(f"\n>>> ContinualLearner: Updating adaptive states for task {self.context_count + 1}...")
        self.prev_theta_p = {n: p.clone().detach() for n, p in self.named_parameters() if
                             n in self.regularized_param_names}
        if self.prev_theta_p is not None and dataset is not None:
            temp_model_state = {n: p.clone().detach() for n, p in self.named_parameters()}
            optimizer_state = self.optimizer.state_dict() if self.optimizer else None
            self.load_state_dict(self.prev_theta_p, strict=False)
            self.prev_H_p = self.compute_hessian(dataset, batch_mode=False)
            self.load_state_dict(temp_model_state, strict=False)
            if optimizer_state:
                self.optimizer.load_state_dict(optimizer_state)
        self.current_H_p = None
        self.context_count += 1
        print(">>> ContinualLearner: Adaptive states updated.")
    '''

    def update_after_task(self, finished_dataset):
        """在一个任务训练结束后调用。保存当前模型状态并计算 prev_H_p。"""
        if not self.use_adaptive:
            return
        print(f"\n>>> ContinualLearner: Updating states after finishing task {self.context_count + 1}...")
        self.prev_theta_p = {n: p.clone().detach() for n, p in self.named_parameters() if
                             n in self.regularized_param_names}
        # self.prev_optimizer_state = self.optimizer.state_dict() if self.optimizer else None # 优化器状态可以按需保留
        self.prev_H_p = self.compute_hessian(finished_dataset, batch_mode=False)
        self.context_count += 1
        print(">>> ContinualLearner: States updated after task.")

    def prepare_for_new_task(self, new_dataset):
        """在新任务训练开始前调用。在新数据集上计算 current_H_p。"""
        if not self.use_adaptive or self.prev_theta_p is None:
            return
        print(f">>> ContinualLearner: Preparing for new task {self.context_count + 1}...")
        # 使用一个临时模型来加载旧参数，避免污染当前模型状态
        temp_model = copy.deepcopy(self)
        temp_model.load_state_dict(self.prev_theta_p, strict=False)
        # 在旧模型上，用新任务的数据计算 current_H_p
        self.current_H_p = temp_model.compute_hessian(new_dataset, batch_mode=False)
        del temp_model
        torch.cuda.empty_cache()
        print(">>> ContinualLearner: Ready for new task.")

    # ----------------- Geometry mismatch attack helpers -----------------#

    def store_gma_reference(self, dataset, allowed_classes=None):
        """Store θ*_1 and its Fisher/Hessian proxy for the geometry mismatch attack."""
        if not self.gma_attack:
            return
        importance_dict = self._gma_snapshot_importance(dataset, allowed_classes)
        if importance_dict is None:
            return
        self.gma_reference_params = {
            n: p.detach().clone() for n, p in self.named_parameters() if p.requires_grad
        }
        self.gma_reference_importance = importance_dict
        # Reset the short window that feeds the λ̄ metric so Task 2 alone determines the mean.
        self.lambda_t_history_since_reference = []

    def geometry_mismatch_adv_loss(self):
        if (not self.gma_attack or self.gma_reference_params is None or
                self.gma_reference_importance is None):
            return None
        losses = []
        for n, p in self.named_parameters():
            if (not p.requires_grad) or (n not in self.gma_reference_params) or (
                    n not in self.gma_reference_importance):
                continue
            ref = self.gma_reference_params[n].to(p.device)
            importance = self.gma_reference_importance[n].to(p.device)
            losses.append((importance * (p - ref) ** 2).sum())
        if len(losses) == 0:
            return None
        adv_loss = -sum(losses)
        self.gma_adv_loss_history.append(float(adv_loss.detach().cpu()))
        return adv_loss

    def _gma_snapshot_importance(self, dataset, allowed_classes):
        importance = {}
        if self.importance_weighting == 'si':
            importance = self._clone_importance_buffers('_SI_omega')
        elif self.importance_weighting == 'mas':
            importance = self._clone_importance_buffers('_MAS_omega')
        elif self.importance_weighting == 'rwalk':
            importance = self._clone_importance_buffers('_RWALK_fisher')
        elif self.importance_weighting == 'fisher':
            # Prefer the Fisher that was just computed for the regulariser, but fall back to a
            # freshly-estimated proxy when running attack-only baselines (e.g., DGR).
            importance = self._clone_importance_buffers('_EWC_estimated_fisher')

        if len(importance) == 0:
            fisher_dict = self.estimate_fisher(dataset, allowed_classes=allowed_classes, return_fisher=True)
            for n, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                fisher_key = n.replace('.', '__')
                if fisher_key in fisher_dict:
                    importance[n] = fisher_dict[fisher_key].detach().clone().to(p.device)

        if len(importance) == 0:
            return None
        return importance

    def _clone_importance_buffers(self, suffix):
        importance = {}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            key = n.replace('.', '__')
            buffer_name = f'{key}{suffix}'
            if hasattr(self, buffer_name):
                importance[n] = getattr(self, buffer_name).detach().clone().to(p.device)
        return importance

    #----------------- XdG-specifc functions -----------------#

    def apply_XdGmask(self, context):
        '''Apply context-specific mask, by setting activity of pre-selected subset of nodes to zero.

        [context]   <int>, starting from 1'''

        assert self.mask_dict is not None
        torchType = next(self.parameters()).detach()

        # Loop over all buffers for which a context-specific mask has been specified
        for i,excit_buffer in enumerate(self.excit_buffer_list):
            gating_mask = np.repeat(1., len(excit_buffer))
            gating_mask[self.mask_dict[context][i]] = 0.    # -> find context-specific mask
            excit_buffer.set_(torchType.new(gating_mask))   # -> apply this mask

    def reset_XdGmask(self):
        '''Remove context-specific mask, by setting all "excit-buffers" to 1.'''
        torchType = next(self.parameters()).detach()
        for excit_buffer in self.excit_buffer_list:
            gating_mask = np.repeat(1., len(excit_buffer))  # -> define "unit mask" (i.e., no masking at all)
            excit_buffer.set_(torchType.new(gating_mask))   # -> apply this unit mask


    #------------- "Synaptic Intelligence"-specifc functions -------------#

    def register_starting_param_values(self):
        '''Register the starting parameter values into the model as a buffer.'''
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    self.register_buffer('{}_SI_prev_context'.format(n), p.detach().clone())
                    self.register_buffer('{}_SI_omega'.format(n), torch.zeros_like(p.detach()))

    def prepare_importance_estimates_dicts(self):
        '''Prepare <dicts> to store running importance estimates and param-values before update.'''
        W = {}
        p_old = {}
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()
        return W, p_old

    def update_importance_estimates(self, W, p_old):
        '''Update the running parameter importance estimates in W.'''
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        W[n].add_(-p.grad*(p.detach()-p_old[n]))
                    p_old[n] = p.detach().clone()

    def update_omega(self, W, epsilon):
        '''After completing training on a context, update the per-parameter regularization strength.

        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed context
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')

                    # Find/calculate new values for quadratic penalty on parameters
                    p_prev = getattr(self, '{}_SI_prev_context'.format(n))
                    p_current = p.detach().clone()
                    p_change = p_current - p_prev
                    omega_add = W[n]/(p_change**2 + epsilon)
                    try:
                        omega = getattr(self, '{}_SI_omega'.format(n))
                    except AttributeError:
                        omega = p.detach().clone().zero_()
                    omega_new = omega + omega_add

                    # Store these new values in the model
                    self.register_buffer('{}_SI_prev_context'.format(n), p_current)
                    self.register_buffer('{}_SI_omega'.format(n), omega_new)

    def surrogate_loss(self):
        '''Calculate SI's surrogate loss.'''
        try:
            losses = []
            for gen_params in self.param_list:
                for n, p in gen_params():
                    if p.requires_grad:
                        # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                        n = n.replace('.', '__')
                        prev_values = getattr(self, '{}_SI_prev_context'.format(n))
                        omega = getattr(self, '{}_SI_omega'.format(n))
                        # Calculate SI's surrogate loss, sum over all parameters
                        losses.append((omega * (p-prev_values)**2).sum())
            return sum(losses)
        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())

    # ----------------- MAS-specifc functions -----------------#
    def initialize_mas_buffers(self):
        '''为MAS初始化缓冲区。'''
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    self.register_buffer('{}_MAS_prev_context'.format(n), p.detach().clone())
                    self.register_buffer('{}_MAS_omega'.format(n), torch.zeros_like(p))

    def update_mas_omega(self, dataset, batch_size=256):
        '''任务结束后，计算并更新每个参数的MAS重要性权重(omega)。'''
        # 1. 初始化一个新的omega累加器
        new_omega = {}
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    new_omega[n] = torch.zeros_like(p)

        # 2. 设置模型为评估模式
        self.eval()
        data_loader = get_data_loader(dataset, batch_size=batch_size, cuda=self._is_on_cuda())

        # 3. 遍历数据集，累积重要性权重
        for x, _ in data_loader:
            x = x.to(self._device())
            self.zero_grad()
            y_hat = self(x)
            l2_norm_sq = torch.norm(y_hat, p=2, dim=1).pow(2)
            loss = l2_norm_sq.mean()
            loss.backward()

            with torch.no_grad():
                for n, p in self.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        n_ = n.replace('.', '__')
                        if n_ in new_omega:
                            new_omega[n_].add_(p.grad.abs())

        # 4. 更新模型的omega缓冲区 (累加)
        with torch.no_grad():
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n_ = n.replace('.', '__')
                    if n_ in new_omega:
                        # 获取旧的omega (如果存在)
                        try:
                            old_omega = getattr(self, '{}_MAS_omega'.format(n_))
                        except AttributeError:
                            old_omega = torch.zeros_like(p)
                        # 将新计算的重要性除以数据集大小后累加
                        updated_omega = old_omega + (new_omega[n_] / len(dataset))
                        self.register_buffer('{}_MAS_omega'.format(n_), updated_omega)
                        # 更新参数快照
                        self.register_buffer('{}_MAS_prev_context'.format(n_), p.detach().clone())

        self.train()

    def mas_loss(self):
        '''计算MAS的正则化损失。'''
        try:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n_ = n.replace('.', '__')
                    omega = getattr(self, '{}_MAS_omega'.format(n_))
                    prev_values = getattr(self, '{}_MAS_prev_context'.format(n_))
                    losses.append((omega * (p - prev_values) ** 2).sum())
            return sum(losses)
        except AttributeError:
            return torch.tensor(0., device=self._device())

    # ----------------- RWALK-specifc functions -----------------#
    def initialize_rwalk_buffers(self):
        '''为RWALK初始化缓冲区。'''
        for n, p in self.named_parameters():
            if p.requires_grad:
                n_ = n.replace('.', '__')
                self.register_buffer('{}_RWALK_prev_context'.format(n_), p.detach().clone())
                # RWALK同时需要Fisher和Score
                self.register_buffer('{}_RWALK_fisher'.format(n_), torch.zeros_like(p))
                self.register_buffer('{}_RWALK_score'.format(n_), torch.zeros_like(p))

    def update_rwalk_fisher(self, dataset):
        '''任务结束后，更新RWALK的Fisher和Score。'''
        # 1. 计算当前任务的Fisher信息 (与EWC相同)
        est_fisher_info = self.estimate_fisher(dataset, return_fisher=True)

        # 2. 更新RWALK的Fisher和Score
        with torch.no_grad():
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n_ = n.replace('.', '__')
                    if n_ in est_fisher_info:
                        # 获取旧的Fisher和Score
                        try:
                            old_fisher = getattr(self, '{}_RWALK_fisher'.format(n_))
                            old_score = getattr(self, '{}_RWALK_score'.format(n_))
                        except AttributeError:
                            old_fisher = torch.zeros_like(p)
                            old_score = torch.zeros_like(p)

                        # RWALK的更新规则
                        p_prev = getattr(self, '{}_RWALK_prev_context'.format(n_))
                        p_current = p.detach().clone()
                        p_change = p_current - p_prev

                        # 更新Score
                        new_score = old_score + 0.5 * (est_fisher_info[n_] + old_fisher) * p_change.pow(2)

                        # 更新Fisher
                        new_fisher = self.gamma * old_fisher + est_fisher_info[n_]

                        # 存回缓冲区
                        self.register_buffer('{}_RWALK_fisher'.format(n_), new_fisher)
                        self.register_buffer('{}_RWALK_score'.format(n_), new_score)
                        self.register_buffer('{}_RWALK_prev_context'.format(n_), p_current)

    def rwalk_loss(self):
        '''计算RWALK的正则化损失。'''
        try:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n_ = n.replace('.', '__')
                    fisher = getattr(self, '{}_RWALK_fisher'.format(n_))
                    score = getattr(self, '{}_RWALK_score'.format(n_))
                    prev_values = getattr(self, '{}_RWALK_prev_context'.format(n_))

                    # RWALK损失 = EWC项 + Score项
                    ewc_term = fisher * (p - prev_values) ** 2
                    loss = ewc_term + 2 * score
                    losses.append(loss.sum())
            return (1. / 2) * sum(losses)
        except AttributeError:
            return torch.tensor(0., device=self._device())


    #----------------- EWC-specifc functions -----------------#

    def initialize_fisher(self):
        '''Initialize diagonal fisher matrix with the prior precision (as in NCL).'''
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    # -take initial parameters as zero for regularization purposes
                    self.register_buffer('{}_EWC_prev_context'.format(n), p.detach().clone()*0)
                    # -precision (approximated by diagonal Fisher Information matrix)
                    self.register_buffer( '{}_EWC_estimated_fisher'.format(n), torch.ones(p.shape) / self.data_size)

    def estimate_fisher(self, dataset, allowed_classes=None, return_fisher=False):
        '''After completing training on a context, estimate diagonal of Fisher Information matrix.

        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''

        # Prepare <dict> to store estimated Fisher Information matrix
        est_fisher_info = {}
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1 (unless specifically asked to do otherwise)
        data_loader = get_data_loader(dataset, batch_size=1 if self.fisher_batch is None else self.fisher_batch,
                                      cuda=self._is_on_cuda())

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for index,(x,y) in enumerate(data_loader):
            # break from for-loop if max number of samples has been reached
            if self.fisher_n is not None:
                if index > self.fisher_n:
                    break
            # run forward pass of model
            x = x.to(self._device())
            output = self(x) if allowed_classes is None else self(x)[:, allowed_classes]
            # calculate FI-matrix (according to one of the four options)
            if self.fisher_labels=='all':
                # -use a weighted combination of all labels
                with torch.no_grad():
                    label_weights = F.softmax(output, dim=1)  # --> get weights, which shouldn't have gradient tracked
                for label_index in range(output.shape[1]):
                    label = torch.LongTensor([label_index]).to(self._device())
                    negloglikelihood = F.cross_entropy(output, label)  #--> get neg log-likelihoods for this class
                    # Calculate gradient of negative loglikelihood
                    self.zero_grad()
                    negloglikelihood.backward(retain_graph=True if (label_index+1)<output.shape[1] else False)
                    # Square gradients and keep running sum (using the weights)
                    for gen_params in self.param_list:
                        for n, p in gen_params():
                            if p.requires_grad:
                                n = n.replace('.', '__')
                                if p.grad is not None:
                                    est_fisher_info[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)
                                if self.randomize_fisher:
                                    idx = torch.randperm(est_fisher_info[n].nelement())
                                    est_fisher_info[n] = est_fisher_info[n].view(-1)[idx].view(
                                        est_fisher_info[n].size())
            else:
                # -only use one particular label for each datapoint
                if self.fisher_labels=='true':
                    # --> use provided true label to calculate loglikelihood --> "empirical Fisher":
                    label = torch.LongTensor([y]) if type(y)==int else y  #-> shape: [self.fisher_batch]
                    if allowed_classes is not None:
                        label = [int(np.where(i == allowed_classes)[0][0]) for i in label.numpy()]
                        label = torch.LongTensor(label)
                    label = label.to(self._device())
                elif self.fisher_labels=='pred':
                    # --> use predicted label to calculate loglikelihood:
                    label = output.max(1)[1]
                elif self.fisher_labels=='sample':
                    # --> sample one label from predicted probabilities
                    with torch.no_grad():
                        label_weights = F.softmax(output, dim=1)       #--> get predicted probabilities
                    weights_array = np.array(label_weights[0].cpu())   #--> change to np-array, avoiding rounding errors
                    label = np.random.choice(len(weights_array), 1, p=weights_array/weights_array.sum())
                    label = torch.LongTensor(label).to(self._device()) #--> change label to tensor on correct device
                # calculate negative log-likelihood
                negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
                # calculate gradient of negative loglikelihood
                self.zero_grad()
                negloglikelihood.backward()
                # square gradients and keep running sum
                for gen_params in self.param_list:
                    for n, p in gen_params():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                est_fisher_info[n] += p.grad.detach() ** 2
                            if self.randomize_fisher:
                                idx = torch.randperm(est_fisher_info[n].nelement())
                                est_fisher_info[n] = est_fisher_info[n].view(-1)[idx].view(est_fisher_info[n].size())

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}

        # 如果只是返回fisher信息（用于其他算法），不更新模型状态
        if return_fisher:
            self.train(mode=mode)
            return est_fisher_info

        # Store new values in the network
        for gen_params in self.param_list:
            for n, p in gen_params():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    # -mode (=MAP parameter estimate)
                    self.register_buffer('{}_EWC_prev_context{}'.format(n, self.context_count+1 if self.offline else ""),
                                         p.detach().clone())
                    # -precision (approximated by diagonal Fisher Information matrix)
                    if (not self.offline) and hasattr(self, '{}_EWC_estimated_fisher'.format(n)):
                        existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                        est_fisher_info[n] += self.gamma * existing_values
                    self.register_buffer(
                        '{}_EWC_estimated_fisher{}'.format(n, self.context_count+1 if self.offline else ""), est_fisher_info[n]
                    )

        # Increase context-count
        # self.context_count += 1

        # Set model back to its initial mode
        self.train(mode=mode)

    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        try:
            losses = []
            # If "offline EWC", loop over all previous contexts as each context has separate penalty term
            num_penalty_terms = self.context_count if (self.offline and self.context_count>0) else 1
            for context in range(1, num_penalty_terms+1):
                for gen_params in self.param_list:
                    for n, p in gen_params():
                        if p.requires_grad:
                            # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                            n = n.replace('.', '__')
                            mean = getattr(self, '{}_EWC_prev_context{}'.format(n, context if self.offline else ""))
                            fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, context if self.offline else ""))
                            # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                            fisher = fisher if self.offline else self.gamma*fisher
                            # Calculate weight regularization loss
                            losses.append((fisher * (p-mean)**2).sum())
            # Sum the regularization loss from all parameters (and from all contexts, if "offline EWC")
            return (1./2)*sum(losses)
        except AttributeError:
            # Regularization loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=self._device())


    # ----------------- KFAC-specifc functions -----------------#

    def initialize_kfac_fisher(self):
        '''Initialize Kronecker-factored Fisher matrix with the prior precision (as in NCL).'''
        fcE = self.fcE
        classifier = self.classifier

        def initialize_for_fcLayer(layer):
            if not isinstance(layer, fc.layers.fc_layer):
                raise NotImplemented
            linear = layer.linear
            g_dim, a_dim = linear.weight.shape
            abar_dim = a_dim + 1 if linear.bias is not None else a_dim
            A = torch.eye(abar_dim) / np.sqrt(self.data_size)
            G = torch.eye(g_dim) / np.sqrt(self.data_size)
            return {"A": A, "G": G, "weight": linear.weight.data * 0,
                    "bias": None if linear.bias is None else linear.bias.data * 0}

        def initialize():
            est_fisher_info = {}
            for i in range(1, fcE.layers + 1):
                label = f"fcLayer{i}"
                layer = getattr(fcE, label)
                est_fisher_info[label] = initialize_for_fcLayer(layer)
            est_fisher_info["classifier"] = initialize_for_fcLayer(classifier)
            return est_fisher_info

        self.KFAC_FISHER_INFO = initialize()

    def estimate_kfac_fisher(self, dataset, allowed_classes=None):
        """After completing training on a context, estimate KFAC Fisher Information matrix.

        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes
        """

        print('computing kfac fisher')

        fcE = self.fcE
        classifier = self.classifier

        def initialize_for_fcLayer(layer):
            if not isinstance(layer, fc.layers.fc_layer):
                raise NotImplemented
            linear = layer.linear
            g_dim, a_dim = linear.weight.shape
            abar_dim = a_dim + 1 if linear.bias is not None else a_dim
            A = torch.zeros(abar_dim, abar_dim)
            G = torch.zeros(g_dim, g_dim)
            if linear.bias is None:
                bias = None
            else:
                bias = linear.bias.data.clone()
            return {"A": A, "G": G, "weight": linear.weight.data.clone(), "bias": bias}

        def initialize():
            est_fisher_info = {}
            for i in range(1, fcE.layers + 1):
                label = f"fcLayer{i}"
                layer = getattr(fcE, label)
                est_fisher_info[label] = initialize_for_fcLayer(layer)
            est_fisher_info["classifier"] = initialize_for_fcLayer(classifier)
            return est_fisher_info

        def update_fisher_info_layer(est_fisher_info, intermediate, label, layer, n_samples, weight=1):
            if not isinstance(layer, fc.layers.fc_layer):
                raise NotImplemented
            if not hasattr(layer, 'phantom'):
                raise Exception(f"Layer {label} does not have phantom parameters")
            g = layer.phantom.grad.detach()
            G = g[..., None] @ g[..., None, :]
            _a = intermediate[label].detach()
            # Here we do one batch at a time (not ideal)
            assert _a.shape[0] == 1
            a = _a[0]

            if classifier.bias is None:
                abar = a
            else:
                o = torch.ones(*a.shape[0:-1], 1).to(self._device())
                abar = torch.cat((a, o), -1)
            A = abar[..., None] @ abar[..., None, :]
            Ao = est_fisher_info[label]["A"].to(self._device())
            Go = est_fisher_info[label]["G"].to(self._device())
            est_fisher_info[label]["A"] = Ao + weight * A / n_samples
            est_fisher_info[label]["G"] = Go + weight * G / n_samples

        def update_fisher_info(est_fisher_info, intermediate, n_samples, weight=1):
            for i in range(1, fcE.layers + 1):
                label = f"fcLayer{i}"
                layer = getattr(fcE, label)
                update_fisher_info_layer(est_fisher_info, intermediate, label, layer, n_samples, weight=weight)
            update_fisher_info_layer(est_fisher_info, intermediate, "classifier", self.classifier, n_samples,
                                     weight=weight)

        # initialize estimated fisher info
        est_fisher_info = initialize()
        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1 (unless specifically asked to do otherwise)
        data_loader = get_data_loader(dataset, batch_size=1 if self.fisher_batch is None else self.fisher_batch,
                                      cuda=self._is_on_cuda())

        n_samples = len(data_loader) if self.fisher_n is None else self.fisher_n

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for i, (x, y) in enumerate(data_loader):
            # break from for-loop if max number of samples has been reached
            if i > n_samples:
                break
            # run forward pass of model
            x = x.to(self._device())
            _output, intermediate = self(x, return_intermediate=True)
            output = _output if allowed_classes is None else _output[:, allowed_classes]
            # calculate FI-matrix (according to one of the four options)
            if self.fisher_labels=='all':
                # -use a weighted combination of all labels
                with torch.no_grad():
                    label_weights = F.softmax(output, dim=1)  # --> get weights, which shouldn't have gradient tracked
                for label_index in range(output.shape[1]):
                    label = torch.LongTensor([label_index]).to(self._device())
                    negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
                    # Calculate gradient of negative loglikelihood
                    self.zero_grad()
                    negloglikelihood.backward(retain_graph=True if (label_index+1)<output.shape[1] else False)
                    update_fisher_info(est_fisher_info, intermediate, n_samples, weight=label_weights[0][label_index])
            else:
                # -only use one particular label for each datapoint
                if self.fisher_labels == 'true':
                    # --> use provided true label to calculate loglikelihood --> "empirical Fisher":
                    label = torch.LongTensor([y]) if type(y) == int else y  # -> shape: [self.fisher_batch]
                    if allowed_classes is not None:
                        label = [int(np.where(i == allowed_classes)[0][0]) for i in label.numpy()]
                        label = torch.LongTensor(label)
                    label = label.to(self._device())
                elif self.fisher_labels == 'pred':
                    # --> use predicted label to calculate loglikelihood:
                    label = output.max(1)[1]
                elif self.fisher_labels == 'sample':
                    # --> sample one label from predicted probabilities
                    with torch.no_grad():
                        label_weights = F.softmax(output, dim=1)  # --> get predicted probabilities
                    weights_array = np.array(label_weights[0].cpu())  # --> change to np-array, avoiding rounding errors
                    label = np.random.choice(len(weights_array), 1, p=weights_array / weights_array.sum())
                    label = torch.LongTensor(label).to(self._device())  # --> change label to tensor on correct device

                # calculate negative log-likelihood
                negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

                # Calculate gradient of negative loglikelihood
                self.zero_grad()
                negloglikelihood.backward()
                update_fisher_info(est_fisher_info, intermediate, n_samples)


        for label in est_fisher_info:
            An = est_fisher_info[label]["A"].to(self._device())  # new kronecker factor
            Gn = est_fisher_info[label]["G"].to(self._device())  # new kronecker factor
            Ao = self.gamma * self.KFAC_FISHER_INFO[label]["A"].to(self._device())  # old kronecker factor
            Go = self.KFAC_FISHER_INFO[label]["G"].to(self._device())               # old kronecker factor

            As, Gs = additive_nearest_kf({"A": Ao, "G": Go}, {"A": An, "G": Gn})  # sum of kronecker factors
            self.KFAC_FISHER_INFO[label]["A"] = As
            self.KFAC_FISHER_INFO[label]["G"] = Gs

            for param_name in ["weight", "bias"]:
                p = est_fisher_info[label][param_name].to(self._device())
                self.KFAC_FISHER_INFO[label][param_name] = p

        # Set model back to its initial mode
        self.train(mode=mode)


    def ewc_kfac_loss(self):
        fcE = self.fcE

        def loss_for_layer(label, layer):
            if not isinstance(layer, fc.layers.fc_layer):
                raise NotImplemented
            info = self.KFAC_FISHER_INFO[label]
            A = info["A"].detach().to(self._device())
            G = info["G"].detach().to(self._device())
            bias0 = info["bias"]
            weight0 = info["weight"]
            bias = layer.linear.bias
            weight = layer.linear.weight
            if bias0 is not None and bias is not None:
                p = torch.cat([weight, bias[..., None]], -1)
                p0 = torch.cat([weight0, bias0[..., None]], -1)
            else:
                p = weight
                p0 = weight0
            assert p.shape[-1] == A.shape[1]
            assert p0.shape[-1] == A.shape[1]
            dp = p.to(self._device()) - p0.to(self._device())
            return torch.sum(dp * (G @ dp @ A))

        classifier = self.classifier
        if self.context_count > 0:
            l = loss_for_layer("classifier", classifier)
            for i in range(1, fcE.layers + 1):
                label = f"fcLayer{i}"
                nl = loss_for_layer(label, getattr(fcE, label))
                l += nl
            return 0.5 * l
        else:
            return torch.tensor(0.0, device=self._device())


    # ----------------- OWM-specifc functions -----------------#

    def estimate_owm_fisher(self, dataset, **kwargs):
        '''After completing training on a context, estimate OWM Fisher Information matrix based on [dataset].'''

        ## QUESTION: Should OWM not also be applied to the outputs??

        fcE = self.fcE
        classifier = self.classifier

        def initialize_for_fcLayer(layer):
            if not isinstance(layer, fc.layers.fc_layer):
                raise NotImplemented
            linear = layer.linear
            g_dim, a_dim = linear.weight.shape
            abar_dim = a_dim + 1 if linear.bias is not None else a_dim
            A = torch.zeros(abar_dim, abar_dim)
            return {'A': A, 'weight': linear.weight.data.clone(),
                    'bias': None if linear.bias is None else linear.bias.data.clone()}

        def initialize():
            est_fisher_info = {}
            for i in range(1, fcE.layers + 1):
                label = f"fcLayer{i}"
                layer = getattr(fcE, label)
                est_fisher_info[label] = initialize_for_fcLayer(layer)
            est_fisher_info['classifier'] = initialize_for_fcLayer(classifier)
            return est_fisher_info

        def update_fisher_info_layer(est_fisher_info, intermediate, label, n_samples):
            _a = intermediate[label].detach()
            # Here we do one batch at a time (not ideal)
            assert (_a.shape[0] == 1)
            a = _a[0]
            if classifier.bias is None:
                abar = a
            else:
                o = torch.ones(*a.shape[0:-1], 1).to(self._device())
                abar = torch.cat((a, o), -1)
            A = abar[..., None] @ abar[..., None, :]
            Ao = est_fisher_info[label]['A'].to(self._device())
            est_fisher_info[label]['A'] = Ao + A / n_samples

        def update_fisher_info(est_fisher_info, intermediate, n_samples):
            for i in range(1, fcE.layers + 1):
                label = f"fcLayer{i}"
                update_fisher_info_layer(est_fisher_info, intermediate, label, n_samples)
            update_fisher_info_layer(est_fisher_info, intermediate, 'classifier', n_samples)

        # initialize estimated fisher info
        est_fisher_info = initialize()
        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1
        data_loader = get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda())

        n_samples = len(data_loader) if self.fisher_n is None else self.fisher_n

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for i, (x, _) in enumerate(data_loader):
            if i > n_samples:
                break
            # run forward pass of model
            x = x.to(self._device())
            output, intermediate = self(x, return_intermediate=True)
            # update OWM importance matrix
            self.zero_grad()
            update_fisher_info(est_fisher_info, intermediate, n_samples)

        if self.context_count == 0:
            self.KFAC_FISHER_INFO = {}

        for label in est_fisher_info:
            An = est_fisher_info[label]['A'].to(self._device())  # new kronecker factor
            if self.context_count == 0:
                self.KFAC_FISHER_INFO[label] = {}
                As = An
            else:
                Ao = self.gamma * self.KFAC_FISHER_INFO[label]['A'].to(self._device())  # old kronecker factor
                frac = 1 / (self.context_count + 1)
                As = (1 - frac) * Ao + frac * An

            self.KFAC_FISHER_INFO[label]['A'] = As

            for param_name in ['weight', 'bias']:
                p = est_fisher_info[label][param_name].to(self._device())
                self.KFAC_FISHER_INFO[label][param_name] = p

        self.context_count += 1

        # Set model back to its initial mode
        self.train(mode=mode)