# trainer.py

import torch
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


# 从我们自己的文件导入
from Model.Diffusion_Model.diffusion_with_condition import GaussianDiffusion1D
from .Utils import cycle, num_to_groups, has_int_squareroot, exists

# denoising_diffusion_pytorch.version import __version__
# 这行可以替换为一个固定的字符串，或者直接删除
__version__ = '1.0.0' 

# --- Data ---
class Dataset1D(Dataset):
    def __init__(self, tensor: torch.Tensor, conditions: torch.Tensor = None, validity_masks: torch.Tensor = None):
        super().__init__()
        self.tensor = tensor.clone()
        self.conditions = conditions.clone() if conditions is not None else None
        self.validity_masks = validity_masks.clone() if validity_masks is not None else None

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        item = [self.tensor[idx].clone()]
        if self.conditions is not None:
            item.append(self.conditions[idx].clone())
        if self.validity_masks is not None:
            item.append(self.validity_masks[idx].clone())
        
        return tuple(item)

# --- Trainer Class ---
# class Trainer1D(object):
#     def __init__(
#         self,
#         diffusion_model: GaussianDiffusion1D,
#         dataset: Dataset,
#         *,
#         ...
#     ):
#         # ... (代码不变)

#     # ... (所有 save, load, train 等方法)



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
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
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
        #self.scheduler = CosineAnnealingLR(self.opt, T_max=train_num_steps, eta_min=1e-6)  # 新增：学习率调度器
        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0
        self.loss_history = [] # 新增：用于记录损失历史

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)


    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    batch_data = next(self.dl)
                    
                    # 动态处理 batch_data
                    data = batch_data[0].to(device)
                    conditions = batch_data[1].to(device) if len(batch_data) > 1 else None
                    masks = batch_data[2].to(device) if len(batch_data) > 2 else None

                    with self.accelerator.autocast():
                        loss = self.model(data, cond_input=conditions, mask=masks, global_step=self.step)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.wait_for_everyone() # 等待所有进程梯度计算完毕
                
                # 记录平均损失
                avg_loss = total_loss / self.gradient_accumulate_every
                self.loss_history.append(avg_loss)
                pbar.set_description(f'loss: {avg_loss:.4f}')

                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                #self.scheduler.step()  # 新增：更新学习率
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            all_samples = self.ema.ema_model.sample(
                            batch_size=self.num_samples,
                            cond_input=self.eval_conditions, # 告诉导航：“现在，执行纸上的这些任务”
                            mask=self.eval_masks)

                        torch.save(all_samples, str(self.results_folder / f'sample-{milestone}.pt'))
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
