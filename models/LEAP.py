import os
import argparse

# import pyrender
import torch
import trimesh
import pickle
import numpy as np

from leap import LEAPBodyModel

def loc(points, device):

    vert = points.detach().cpu().numpy()

    bb_min, bb_max = np.min(vert, axis=1, keepdims=True), np.max(vert, axis=1, keepdims=True)

    x = ((bb_min[:, 0, 0] + bb_max[:, 0, 0]) / 2.).reshape(-1, 1)
    y = ((bb_min[:, 0, 1] + bb_max[:, 0, 1]) / 2.).reshape(-1, 1)
    z = ((bb_min[:, 0, 2] + bb_max[:, 0, 2]) / 2.).reshape(-1, 1)

    loc = np.hstack([x, y, z])

    return torch.from_numpy(loc).to(device=device, dtype=torch.float32)

def normalize_for_leap(points, leap_target, device):

    p_loc = loc(points, device)
    t_loc = loc(leap_target, device)

    print("p_loc shape", p_loc.shape)
    print("t_loc shape", t_loc.shape)
    diff = (t_loc - p_loc).unsqueeze(dim=1)
    diff = diff.repeat([1, points.shape[1], 1])
    aligned = (points + diff).to(dtype=torch.float32)
    return aligned, diff

def query_leap(points, leap_path, smpl_body, bm_path, batch_size, device, canonical_points=False, vis=False, use_pkl=True):

    smpl_body = {key: val.to(device=device) if torch.is_tensor(val) else val for key, val in smpl_body.items()}

    # load LEAP
    print("batch size", batch_size)
    leap_model = LEAPBodyModel(leap_path,
                               bm_path=os.path.join(bm_path, smpl_body['gender'], 'model.pkl'),
                               num_betas=smpl_body['betas'].shape[1],
                               batch_size=batch_size,
                               device=device)

    leap_model = leap_model.float()

    # Set LEAP params
    leap_model.set_parameters(betas=smpl_body['betas'].to(dtype=torch.float32),
                              pose_body=smpl_body['pose_body'].to(dtype=torch.float32),
                              pose_hand=smpl_body['pose_hand'].to(dtype=torch.float32))

    leap_model.forward_parametric_model()

    points_normal, diff = normalize_for_leap(points, leap_model.posed_vert, device)
    point_weights, can_points = leap_model.model.inv_lbs(points_normal, leap_model.can_vert, leap_model.posed_vert, leap_model.fwd_transformation, compute_can_points=True)
    can_points_orig = can_points - diff

    return point_weights, can_points_orig
