import numpy as np
import torch
import os
import random
import argparse
from tqdm import tqdm
from sklearn.metrics import classification_report

from data.dataset import ShapeDataset
from models.transformer_base import PCTransformer


def run_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            pc, y = data[0].to(device), data[1].to(device)
            logits = model(pc)
            y_true += y.data.cpu().numpy().tolist()
            y_pred += torch.argmax(logits, dim=1).data.cpu().numpy().tolist()
    print('Classification report for {} set'.format(dataloader.dataset.phase))
    target_names = list(dataloader.dataset.class_map.keys())
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    
def test(args):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    dataset_t = ShapeDataset('test', args)
    dataloader_t = torch.utils.data.DataLoader(dataset_t,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               pin_memory=torch.cuda.is_available()
                                               )
    dataset_v = ShapeDataset('valid', args)
    dataloader_v = torch.utils.data.DataLoader(dataset_v,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.num_workers,
                                               drop_last=False,
                                               pin_memory=torch.cuda.is_available()
                                               )

    model = PCTransformer(args)
    checkpoint = torch.load(args.weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    run_model(model, dataloader_t, device)
    run_model(model, dataloader_v, device)

    
if __name__ == "__main__":
    args = argparse.Namespace(
        seed=111,
        num_workers=4,
        data_path='./dataset-v2',
        weights_path='./last.pt',
        num_cls=6,
        n_points_mesh=25000,
        n_points_batch=1024,
        batch_size=110,
        hid_dim=128,
        nhead=2,
        dropout=0.4,
        dim_fc=1024,
        n_attn=4,
    )
    seed_value = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

    test(args)
