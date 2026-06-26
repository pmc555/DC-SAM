import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, StepLR, SequentialLR
from pareconv.engine import EpochBasedTrainer
from config import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss, Evaluator
from pareconv.utils.common import print_model_parameters


# 训练稳定性默认值；如需暴露给 config 再升级到 cfg.optim 字段
WARMUP_EPOCHS = 5   # 5-epoch warmup → lr: 1e-5 ramps to 1e-4 over 5 epochs (prevents 10× lr jump)
GRAD_CLIP_NORM = 1.0
UNCERTAINTY_LR_FACTOR = 0.1  # uncertainty weights train at 0.1× model lr to avoid log_s divergence


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg, max_epoch=cfg.optim.max_epoch, run_grad_check=False, autograd_anomaly_detection=False)

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
        loading_time = time.time() - start_time
        self.logger.info('Data loader created: {:.3f}s collapsed.'.format(loading_time))
        self.logger.info('Calibrate neighbors: {}.'.format(neighbor_limits))
        self.register_loader(train_loader, val_loader)

        # model, optimizer, scheduler
        model = create_model(cfg).cuda()
        model = self.register_model(model)
        print_model_parameters(model)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
        )
        self.register_optimizer(optimizer)

        # warmup (LinearLR over WARMUP_EPOCHS) → StepLR decay
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
        decay = StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[WARMUP_EPOCHS])
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

        # 将 OverallLoss 的可学习不确定性权重（log_s_coarse/ri/re）加入优化器
        # 不加 weight_decay 避免正则化压制方差估计
        loss_params = [p for p in self.loss_func.parameters() if p.requires_grad]
        if loss_params:
            # Use a small, fixed lr for log_s_* uncertainty weights so they don't
            # diverge when task losses are large (gradient ∝ -exp(-log_s)*L can be
            # very large when L≈5 and lr is at its peak value).
            uncertainty_lr = cfg.optim.lr * UNCERTAINTY_LR_FACTOR
            optimizer.add_param_group({'params': loss_params, 'lr': uncertainty_lr, 'weight_decay': 0.0})
            self.logger.info(f'Added {len(loss_params)} uncertainty-weight params to optimizer (lr={uncertainty_lr:.2e}).')

    def train_step(self, epoch, iteration, data_dict):
        # AMP: model forward 在 autocast 下用 fp16 加速
        with torch.autocast('cuda', enabled=self.use_amp):
            output_dict = self.model(data_dict)
        # loss / evaluator 在 fp32 下执行：
        # 将模型输出中的 fp16 浮点张量统一转为 fp32，避免与 data_dict 中
        # fp32 张量（来自 dataloader）在 index_put / 距离计算中发生 dtype 不匹配
        output_fp32 = {
            k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in output_dict.items()
        }
        loss_dict = self.loss_func(output_fp32, data_dict)
        result_dict = self.evaluator(output_fp32, data_dict)
        loss_dict.update(result_dict)
        return output_fp32, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

    def after_backward(self, epoch, iteration, data_dict, output_dict, result_dict):
        # AMP: unscale 后再 clip，保证 clip_norm 作用于真实梯度而非缩放后的梯度
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        # 同时 clip model + loss_func（后者含 log_s_* 参数）
        all_params = list(self.model.parameters()) + list(self.loss_func.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=GRAD_CLIP_NORM)

    def after_train_epoch(self, epoch):
        # 课程学习：每 epoch 结束后收紧 fine loss 判定半径
        self.loss_func.anneal(epoch, self.max_epoch)


def main():
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run()



if __name__ == '__main__':
    main()