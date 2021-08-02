import numpy as np
import pandas as pd
import os
import torch
import argparse
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st

from transformer_base import PCTransformer


def get_pc(verts, faces):
    mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
    pc = sample_points_from_meshes(mesh, 25000)
    idx = torch.randperm(pc.size(1))[:1024]
    pc = pc[:, idx, :]
    pc = (pc - pc.mean(dim=1)) / torch.max(pc)
    return pc


def prepare_model():
    args = argparse.Namespace(
        num_cls=6,
        hid_dim=128,
        nhead=2,
        dropout=0.4,
        dim_fc=1024,
        n_attn=4,
    )
    model = PCTransformer(args)
    checkpoint = torch.load('./last.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


st.title('3D model visualization')
if 'model' not in st.session_state:
    st.session_state.model = prepare_model()
path = st.text_input('path to model')
if len(path) > 0:
    if os.path.exists(path):
        try:
            verts, faces, _ = load_obj(path)
            pc = get_pc(verts, faces)
            with torch.no_grad():
                score = F.softmax(st.session_state.model(pc), dim=-1).cpu().squeeze().numpy()

            verts = (verts - verts.mean(0)) / torch.max(verts)
            x, y, z = verts.numpy().T
            I, J, K = faces.verts_idx.numpy().T
            mesh = go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=I,
                j=J,
                k=K,
                colorbar_title='z',
                intensitymode='cell',
                showscale=True)
            mesh.update(lighting=dict(ambient=0.18,
                                      diffuse=1,
                                      fresnel=.1,
                                      specular=1,
                                      roughness=.1),
                        lightposition=dict(x=100,
                                           y=200,
                                           z=150))

            fig = go.Figure(data=[mesh])
            st.plotly_chart(fig, use_container_width=True)
            df = pd.DataFrame({'shape': ['Cone', 'Cube', 'Cylinder', 'Plane', 'Torus', 'Sphere'], 'score': score})
            fig = px.bar(df, x='shape', y='score', title='Classification score')
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.write('Not 3D model .obj file')
    else:
        st.write('No such file or directory')
else:
    pass
