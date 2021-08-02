import os
import copy
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, smoothing=0.0, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def smooth_one_hot(self, targets, n_classes):
        assert 0 <= self.smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(self.smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - self.smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = self.smooth_one_hot(targets, inputs.size(-1))
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class LrScheduler:
    def __init__(self, n_steps, args):
        self.warm_len = n_steps * args.warmup_steps_part
        self.dec_len = n_steps - self.warm_len
        self.lr_warm_step = args.lr_peak / self.warm_len
        self.lr_dec_step = args.lr_peak / self.dec_len
        self.done_steps = 0
        self._step = 0
        self._lr = 0

    def step(self, optimizer):
        self._step += 1
        lr = self.learning_rate()
        for p in optimizer.param_groups:
            p['lr'] = lr

    def learning_rate(self, step=None):
        if step is None:
            step = self._step
        if step <= self.warm_len:
            delta = self.lr_warm_step * (step - self.done_steps)
            self._lr += delta
        else:
            delta = self.lr_dec_step * (step - self.done_steps)
            self._lr -= delta
        self.done_steps = step
        return self._lr

    def state_dict(self):
        sd = copy.deepcopy(self.__dict__)
        return sd

    def load_state_dict(self, sd):
        for k in sd.keys():
            self.__setattr__(k, sd[k])


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_net(model, lr_scheduler, optimizer, epoch, path, args):
    state_dict = {}
    save_fname = '%s.pt' % (epoch)
    save_path = os.path.join(path, save_fname)
    if len(args.gpu_ids) > 1:
        state_dict['model_state_dict'] = model.module.cpu().state_dict()
        model.cuda()
    elif torch.cuda.is_available():
        state_dict['model_state_dict'] = model.cpu().state_dict()
        model.cuda()
    else:
        state_dict['model_state_dict'] = model.state_dict()
    state_dict['optimizer_state_dict'] = optimizer.state_dict()
    state_dict['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    print(f'Saved checkpoint to {save_path}')
    torch.save(state_dict, save_path)