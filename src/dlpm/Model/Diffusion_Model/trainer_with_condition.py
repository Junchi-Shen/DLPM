# trainer.py

import torch
import numpy as np
import random as python_random
import os
import math
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader


from pathlib import Path
from random import random
from functools import partial
from multiprocessing import cpu_count

from accelerate import Accelerator
from ema_pytorch import EMA
from tqdm.auto import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LambdaLR


# 从我们自己的文件导入
from Model.Diffusion_Model.diffusion_with_condition import GaussianDiffusion1D
from .Utils import cycle, num_to_groups, has_int_squareroot, exists

# denoising_diffusion_pytorch.version import __version__
# 这行可以替换为一个固定的字符串，或者直接删除
__version__ = '1.0.0' 


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


# --- Data ---
class Dataset1D(Dataset):
    def __init__(
        self,
        tensor: torch.Tensor,
        conditions: torch.Tensor = None,
        validity_masks: torch.Tensor = None,
        sample_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.tensor = tensor.clone()
        self.conditions = conditions.clone() if conditions is not None else None
        self.validity_masks = validity_masks.clone() if validity_masks is not None else None
        self.sample_weights = sample_weights.clone() if sample_weights is not None else None

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        item = [self.tensor[idx].clone()]
        if self.conditions is not None:
            item.append(self.conditions[idx].clone())
        if self.validity_masks is not None:
            item.append(self.validity_masks[idx].clone())
        if self.sample_weights is not None:
            item.append(self.sample_weights[idx].clone())
        
        return tuple(item)


class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        warmup_ratio = 0.0,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        checkpoint_every = None,
        sample_every = None,
        num_samples = 25,
        results_folder = './trainning_save',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps
        self.checkpoint_every = checkpoint_every
        self.sample_every = sample_every if sample_every is not None else save_and_sample_every

       
        # dataset and dataloader

        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 1)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        
        if self.accelerator.is_main_process:
            self.eval_batch = next(self.dl)
            # 如果数据集返回了多个元素 (data, cond, mask)
            if len(self.eval_batch) > 1:
                # 将评估用的条件和掩码移动到设备上并保存
                self.eval_conditions = self.eval_batch[1][:self.num_samples].to(self.device)
                if len(self.eval_batch) > 2:
                    self.eval_masks = self.eval_batch[2][:self.num_samples].to(self.device)
                else:
                    self.eval_masks = None
            else:
                self.eval_conditions = None
                self.eval_masks = None


        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        warmup_ratio = float(warmup_ratio)
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError(f'warmup_ratio must be in [0, 1), got {warmup_ratio}')
        self.warmup_steps = int(round(train_num_steps * warmup_ratio))
        if self.warmup_steps > 0:
            min_factor = 0.01
            decay_steps = max(train_num_steps - self.warmup_steps, 1)

            def lr_factor(step):
                if step < self.warmup_steps:
                    return max(min_factor, float(step + 1) / float(self.warmup_steps))
                progress = min(max((step - self.warmup_steps) / decay_steps, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_factor + (1.0 - min_factor) * cosine

            self.scheduler = LambdaLR(self.opt, lr_lambda=lr_factor)
        else:
            self.scheduler = CosineAnnealingLR(self.opt, T_max=train_num_steps, eta_min=train_lr/100)
        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        # step counter state

        self.step = 0
        self.loss_history = [] # 新增：用于记录损失历史
        self.loss_component_history = []

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt, self.scheduler = self.accelerator.prepare(self.model, self.opt, self.scheduler)


    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        numpy_state = np.random.get_state()
        data = {
                'step': self.step,
                'model': self.accelerator.get_state_dict(self.model),
                'opt': self.opt.state_dict(),
                'scheduler': self.scheduler.state_dict(), # 新增保存项
                'ema': self.ema.state_dict(),
                'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
                'loss_history': list(self.loss_history),
                'loss_component_history': list(self.loss_component_history),
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'numpy_rng_state': {
                    'bit_generator': numpy_state[0],
                    'state': torch.from_numpy(numpy_state[1].astype(np.int64)),
                    'pos': int(numpy_state[2]),
                    'has_gauss': int(numpy_state[3]),
                    'cached_gaussian': float(numpy_state[4]),
                },
                'python_rng_state': python_random.getstate(),
                'version': __version__
        }

        atomic_torch_save(data, self.results_folder / f'model-{milestone}.pt')
        ema_model = self.ema.ema_model
        atomic_torch_save(
            ema_model.model.state_dict(),
            self.results_folder / f'unet_conditional_model_ema-{milestone}.pth'
        )
        if getattr(ema_model, 'condition_network', None) is not None:
            atomic_torch_save(
                ema_model.condition_network.state_dict(),
                self.results_folder / f'condition_network_ema-{milestone}.pth'
            )

    def load(self, milestone, source_folder=None):
        accelerator = self.accelerator
        device = accelerator.device

        source_folder = Path(source_folder) if source_folder is not None else self.results_folder
        data = torch.load(str(source_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if 'scheduler' in data:
            self.scheduler.load_state_dict(data['scheduler'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

        self.loss_history = list(data.get('loss_history', []))
        self.loss_component_history = list(data.get('loss_component_history', []))
        if 'torch_rng_state' in data:
            torch.set_rng_state(data['torch_rng_state'].cpu())
        if torch.cuda.is_available() and exists(data.get('cuda_rng_state_all')):
            torch.cuda.set_rng_state_all([state.cpu() for state in data['cuda_rng_state_all']])
        if 'numpy_rng_state' in data:
            np_state = data['numpy_rng_state']
            np.random.set_state((
                np_state['bit_generator'],
                np_state['state'].cpu().numpy().astype(np.uint32),
                int(np_state['pos']),
                int(np_state['has_gauss']),
                float(np_state['cached_gaussian']),
            ))
        if 'python_rng_state' in data:
            python_random.setstate(data['python_rng_state'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        consecutive_bad = 0
        HARD_BAD_LIMIT = 50

        def first_bad_grad_param(model):
            """
            返回第一个出现非有限梯度的参数信息：
            (name, nan_cnt, inf_cnt, max_finite_abs, dtype, shape)
            """
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                finite = torch.isfinite(g)
                if finite.all():
                    continue

                nan_cnt = int(torch.isnan(g).sum().item())
                inf_cnt = int(torch.isinf(g).sum().item())

                # max 只在 finite 上算，避免“全 NaN 时 max 变 0”的假象
                if finite.any():
                    max_finite_abs = float(g[finite].abs().max().item())
                else:
                    max_finite_abs = float("nan")

                return name, nan_cnt, inf_cnt, max_finite_abs, g.dtype, tuple(g.shape)

            return None

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()

                # ✅ 每个 step 开始先清梯度，避免残留污染
                self.opt.zero_grad(set_to_none=True)

                total_loss = 0.0
                bad_group = False
                valid_micro = 0

                # ---------------------------
                # 1) 梯度累积
                # ---------------------------
                for _ in range(self.gradient_accumulate_every):
                    batch_data = next(self.dl)

                    data = batch_data[0].to(device)
                    conditions = batch_data[1].to(device) if len(batch_data) > 1 else None
                    masks = batch_data[2].to(device) if len(batch_data) > 2 else None
                    sample_weights = batch_data[3].to(device) if len(batch_data) > 3 else None

                    with accelerator.autocast():
                        loss = self.model(
                            data,
                            cond_input=conditions,
                            mask=masks,
                            sample_weight=sample_weights,
                            global_step=self.step
                        )

                    # 模型内部拦截返回 None：丢弃整组
                    if loss is None:
                        bad_group = True
                        accelerator.print(f"⚠️ Step {self.step} loss=None，丢弃整组累积")
                        break

                    loss = loss / self.gradient_accumulate_every
                    total_loss += float(loss.detach().item())

                    accelerator.backward(loss)
                    valid_micro += 1

                # ---------------------------
                # 2) 本组失败：直接跳过（并计数）
                # ---------------------------
                if bad_group or valid_micro == 0:
                    self.opt.zero_grad(set_to_none=True)
                    consecutive_bad += 1
                    if consecutive_bad >= HARD_BAD_LIMIT:
                        raise RuntimeError(
                            f"连续 {HARD_BAD_LIMIT} 组累积失败（loss=None 或全失败），请检查数据/模型数值稳定性。")
                    continue

                # ---------------------------
                # 3) 梯度检查：定位到底是哪一层炸
                # ---------------------------
                accelerator.wait_for_everyone()

                bad = first_bad_grad_param(self.model)
                if bad is not None:
                    accelerator.print(
                        f"⚠️ Step {self.step} 梯度含 Inf/NaN -> 丢弃整组 | "
                        f"param={bad[0]} | max|grad|={bad[3]:.3e} | "
                        f"dtype={bad[4]} | shape={bad[5]} | "
                        f"nan={bad[1]} inf={bad[2]}"
                    )
                    self.opt.zero_grad(set_to_none=True)
                    consecutive_bad += 1
                    if consecutive_bad >= HARD_BAD_LIMIT:
                        raise RuntimeError(
                            f"连续 {HARD_BAD_LIMIT} 次出现坏梯度（Inf/NaN）。请进一步定位 loss 分支或模型数值稳定性。")
                    continue

                # ✅ 本步成功：清零连续坏计数
                consecutive_bad = 0

                # ---------------------------
                # 4) 正常更新
                # ---------------------------
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                self.scheduler.step()
                self.opt.zero_grad(set_to_none=True)

                self.step += 1
                pbar.update(1)

                self.loss_history.append(total_loss)
                unwrapped_model = accelerator.unwrap_model(self.model)
                components = getattr(unwrapped_model, 'last_loss_components', None)
                if components and int(components.get('step', -1)) == self.step - 1:
                    self.loss_component_history.append(dict(components))
                    denoising = max(float(components.get('denoising', 0.0)), 1e-12)
                    accelerator.print(
                        '[LOSS COMPONENTS] '
                        f"step={components['step']} "
                        f"denoise={components['denoising']:.6f} "
                        f"financial={components['financial']:.6f} "
                        f"ratio={components['financial'] / denoising:.3f} "
                        f"warmup={components['financial_warmup']:.3f}"
                    )
                pbar.set_description(f"loss: {total_loss:.4f}")

                if accelerator.is_main_process:
                    self.ema.update()

                    if self.checkpoint_every and self.step != 0 and self.step % self.checkpoint_every == 0:
                        milestone = self.step
                        self.save(milestone)

                    if self.sample_every and self.step != 0 and self.step % self.sample_every == 0:
                        self.ema.ema_model.eval()
                        with torch.no_grad():
                            milestone = self.step
                            n = self.num_samples
                            if self.eval_conditions is not None:
                                n = min(n, self.eval_conditions.shape[0])
                            all_samples = self.ema.ema_model.sample(
                                batch_size=n,
                                cond_input=self.eval_conditions[:n] if self.eval_conditions is not None else None,
                                mask=self.eval_masks[:n] if self.eval_masks is not None else None
                            )
                        torch.save(all_samples, str(self.results_folder / f"sample-{milestone}.pt"))

        accelerator.print("training complete")
