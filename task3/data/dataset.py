import os
import glob

import torch
from torch.utils.data import Dataset

from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes


class ShapeDataset(Dataset):
    def __init__(self, phase, args):
        self.class_map = {
            'cone': 0,
            'cube': 1,
            'cylinder': 2,
            'plane': 3,
            'torus': 4,
            'uv_sphere': 5
        }
        self.phase = phase
        self.data = self.load_data(args.data_path, phase)
        self.args = args

    def __getitem__(self, idx):
        path, label = self.data[idx]
        verts, faces, _ = load_obj(path)
        mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
        points = sample_points_from_meshes(mesh, self.args.n_points_mesh)
        points = self.transform(points)
        return points.squeeze(), torch.tensor(label, dtype=torch.int64)

    def transform(self, pc):
        idx = torch.randperm(pc.size(1))[:self.args.n_points_batch]
        pc = pc[:, idx, :]
        pc = (pc - pc.mean(dim=1)) / torch.max(pc)
        if self.phase == 'train':
            pc = self.add_noise(pc)
            pc = self.translate(pc)
        return pc

    def translate(self, pc):
        a = torch.FloatTensor(1, 1, 3).uniform_(0.6, 1.4)
        b = torch.FloatTensor(1, 1, 3).uniform_(-0.1, 0.1)
        return pc * a + b

    def add_noise(self, pc, sigma=0.01, clip=0.02):
        noise = torch.clamp(sigma * torch.rand_like(pc), -clip, clip)
        pc += noise
        return pc

    def __len__(self):
        return len(self.data)

    def load_data(self, path, phase):
        obj_type = os.listdir(path)
        data = []
        for type_ in obj_type:
            cur_path = os.path.join(path, type_, phase)
            obj = glob.glob(os.path.join(cur_path, '*.obj'))
            data += [(p, self.class_map[type_]) for p in obj]
        return data
