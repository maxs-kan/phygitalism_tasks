import numpy as np
import torch
import wandb
import time
import os
import random
import argparse
from sklearn.metrics import accuracy_score

from data.dataset import ShapeDataset
from models.transformer_base import PCTransformer
from utils import LrScheduler, SmoothCrossEntropyLoss, mkdir, save_net


def train(args):
    save_path = os.path.join('./checkpoints', args.name)
    mkdir(save_path)
    torch.cuda.set_device(args.gpu_ids[0])
    device = torch.device('cuda:{}'.format(torch.cuda.current_device()) if torch.cuda.is_available else 'cpu')

    dataset = ShapeDataset('train', args)
    dataset_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             drop_last=False,
                                             pin_memory=torch.cuda.is_available()
                                             )
    print('The number of training data = {}'.format(dataset_size))
    dataset_v = ShapeDataset('test', args)
    dataset_size_v = len(dataset_v)
    dataloader_v = torch.utils.data.DataLoader(dataset_v,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               pin_memory=torch.cuda.is_available()
                                               )
    print('The number of test data = {}'.format(dataset_size_v))

    model = PCTransformer(args)
    model.to(device)
    if len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, args.gpu_ids).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0., weight_decay=args.w_decay)
    n_steps = (dataset_size // args.batch_size + 1) * args.n_epochs
    lr_scheduler = LrScheduler(n_steps, args)
    class_weight = torch.tensor([1., 2., 1., 3.3, 1., 2., ], device=device)
    loss_fn = SmoothCrossEntropyLoss(smoothing=args.smooth, weight=class_weight)

    if not args.debug:
        wandb.init(project="point_cloud", name=args.name)
        wandb.config.update(args)
        wandb.watch(model)

    global_iter = 0
    val_iter = 0
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_start_time = time.time()
        y_true = []
        y_pred = []
        for i, data in enumerate(dataloader):
            iter_start_time = time.time()
            global_iter += 1
            pc, y = data[0].to(device), data[1].to(device)
            logits = model(pc)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            lr_scheduler.step(optimizer)
            loss.backward()
            optimizer.step()
            y_true += y.data.cpu().numpy().tolist()
            y_pred += torch.argmax(logits, dim=1).data.cpu().numpy().tolist()
            iter_finish_time = time.time()
            if global_iter % args.loss_freq == 0:
                print('{} img procesed out of {}'.format((i + 1) * args.batch_size, dataset_size))
                if not args.debug:
                    wandb.log({'loss': loss})
        if not args.debug:
            wandb.log({'train_acc': accuracy_score(y_true, y_pred), 'epoch': epoch})
        print('Validation')
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, data in enumerate(dataloader_v):
                val_iter += 1
                pc, y = data[0].to(device), data[1].to(device)
                logits = model(pc)
                loss = loss_fn(logits, y)
                y_true += y.data.cpu().numpy().tolist()
                y_pred += torch.argmax(logits, dim=1).data.cpu().numpy().tolist()
                if val_iter % args.loss_freq == 0:
                    print('{} img procesed out of {}'.format((i + 1) * args.batch_size, dataset_size_v))
                    if not args.debug:
                        wandb.log({'val_loss': loss, 'val_step': val_iter})
        if not args.debug:
            wandb.log({'val_acc': accuracy_score(y_true, y_pred), 'epoch': epoch})
        if epoch % args.save_epoch_freq == 0:
            print('saving the model at the end of epoch {}, iters {}'.format(epoch, global_iter))
            save_net(model, lr_scheduler, optimizer, epoch, save_path, args)
        print('End of epoch {} / {} \t Time Taken: {:04.2f} sec'.format(epoch, args.n_epochs,
                                                                        time.time() - epoch_start_time))
    save_net(model, lr_scheduler, optimizer, 'last', save_path, args)
    print('Finish')


if __name__ == "__main__":
    args = argparse.Namespace(
        save_epoch_freq=100,
        gpu_ids=[1, 2],
        debug=False,
        name='final',
        seed=111,
        loss_freq=1,
        num_workers=4,
        data_path='./dataset-v2',
        num_cls=6,
        n_points_mesh=25000,
        n_points_batch=1024,
        batch_size=110,
        n_epochs=300,
        warmup_steps_part=0.2,
        smooth=0.05,
        lr_peak=2e-4,
        hid_dim=128,
        nhead=2,
        dropout=0.4,
        w_decay=5e-2,
        dim_fc=1024,
        n_attn=4,
    )
    seed_value = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    train(args)
