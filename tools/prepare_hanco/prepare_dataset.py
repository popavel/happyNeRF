import os
import sys

import json

import torch
import yaml
import pickle
import numpy as np
from tqdm import tqdm

from pathlib import Path

from third_parties.mano.manopth_hanco.manopth.manolayer import ManoLayer

sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))

from absl import app
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    'hanco.yaml',
                    'the path of config file')

MODEL_DIR = '../../third_parties/mano/manopth_hanco/manopth/models_for_manopth'


def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def main(argv):
    del argv  # Unused.

    cfg = parse_config()
    subject = cfg['dataset']['subject']
    hand_type = cfg['dataset']['hand_type']

    dataset_dir = cfg['dataset']['path']
    subject_dir = os.path.join(dataset_dir, subject)
    output_path = subject_dir

    with open(os.path.join(subject_dir, 'metadata.json'), 'r') as f:
        frame_infos = json.load(f)

    # mano_model = MANO(hand_type=hand_type, model_dir=MODEL_DIR)
    mano = ManoLayer(use_pca=False, ncomps=45, side=hand_type,
                     flat_hand_mean=False, center_idx=None, mano_root=MODEL_DIR)

    cameras = {}
    mesh_infos = {}
    all_betas = []
    all_global_t = []

    all_betas_copy = []
    for frame_base_name in tqdm(frame_infos):
        cam_body_info = frame_infos[frame_base_name]
        betas_torch = torch.Tensor(cam_body_info['betas'])
        betas_numpy = betas_torch.detach().cpu().numpy()[0].copy()
        all_betas_copy.append(betas_numpy)
    avg_betas_copy = np.mean(np.stack(all_betas_copy, axis=0), axis=0)
    avg_betas_torch_copy = torch.from_numpy(avg_betas_copy[None, ...].copy())

    global_t_assigned = False
    for frame_base_name in tqdm(frame_infos):
        cam_body_info = frame_infos[frame_base_name]

        # poses = np.array(cam_body_info['poses'], dtype=np.float32)
        # betas = np.array(cam_body_info['betas'], dtype=np.float32)
        # global_t = np.array(cam_body_info['global_t'], dtype=np.float32)

        # this is needed for manolayer
        poses_torch = torch.Tensor(cam_body_info['poses'])
        betas_torch = torch.Tensor(cam_body_info['betas'])
        global_t_torch = torch.Tensor(cam_body_info['global_t'])

        if not global_t_assigned:
            global_t_1 = global_t_torch
            global_t_assigned = True

        # this is needed for humanNeRF
        K = np.array(cam_body_info['cam_intrinsics'], dtype=np.float32)
        E = np.array(cam_body_info['cam_extrinsics'], dtype=np.float32)

        # this format is needed for humanNeRF
        poses_numpy = poses_torch.detach().cpu().numpy()[0].copy()
        betas_numpy = betas_torch.detach().cpu().numpy()[0].copy()
        global_t_numpy = global_t_torch.detach().cpu().numpy()[0, 0, :].copy()

        all_betas.append(betas_numpy)
        all_global_t.append(global_t_numpy)

        ##############################################
        # Below we transfer the global body rotation to camera pose

        # Get T-pose joints
        # _, tpose_joints = mano_model(np.zeros_like(poses), betas)
        # TODO-q: Do we need global_t here, probably yes?
        _, tpose_joints = mano(torch.zeros_like(poses_torch), betas_torch, global_t_torch)
        # _, tpose_joints = mano(torch.zeros_like(poses_torch), avg_betas_torch_copy, global_t_torch)
        # _, tpose_joints = mano(torch.zeros_like(poses_torch), betas_torch)

        # tpose_v, tpose_joints = mano(torch.zeros_like(poses_torch), avg_betas_torch_copy, global_t_torch)
        # tpose_v1, tpose_joints_v1 = mano(torch.zeros_like(poses_torch), betas_torch, global_t_torch)
        #
        # import matplotlib.pyplot as plt
        #
        # tpose_v_np = tpose_v.detach().cpu().numpy()[0].copy()
        # tpose_v1_np = tpose_v1.detach().cpu().numpy()[0].copy()
        # x_v = tpose_v_np[:, 0]
        # y_v = tpose_v_np[:, 1]
        # z_v = tpose_v_np[:, 2]
        #
        # fig = plt.figure()
        #
        # # syntax for 3-D plotting
        # ax = plt.axes(projection='3d')
        #
        # # syntax for plotting
        # ax.scatter(x_v, y_v, z_v, 'green')
        # ax.set_title('')
        # plt.show()

        tpose_joints_numpy = tpose_joints.detach().cpu().numpy()[0].copy()

        # x_j = tpose_joints_numpy[:, 0]
        # y_j = tpose_joints_numpy[:, 1]
        # z_j = tpose_joints_numpy[:, 2]
        # fig = plt.figure()
        #
        # # syntax for 3-D plotting
        # ax = plt.axes(projection='3d')
        #
        # # syntax for plotting
        # ax.plot3D(x_j, y_j, z_j, 'green')
        # ax.set_title('')
        # plt.show()

        # get global Rh, Th
        wrist_pos = tpose_joints_numpy[0].copy()
        Th = wrist_pos
        Rh = poses_numpy[:3].copy()

        # get refined T-pose joints
        tpose_joints_numpy = tpose_joints_numpy - wrist_pos[None, :]

        # remove global rotation from body pose
        poses_numpy[:3] = 0

        # get posed joints using body poses without global rotation
        # _, joints = mano_model(poses, betas)
        poses_torch[:, :3] = 0
        _, joints = mano(poses_torch, betas_torch, global_t_torch)
        # _, joints = mano(poses_torch, avg_betas_torch_copy, global_t_torch)
        # _, joints = mano(poses_torch, betas_torch)
        joints_numpy = joints.detach().cpu().numpy()[0].copy()
        joints_numpy = joints_numpy - wrist_pos[None, :]

        mesh_infos[frame_base_name] = {
            'Rh': Rh,
            'Th': Th,
            'poses': poses_numpy,
            'joints': joints_numpy,
            'tpose_joints': tpose_joints_numpy
        }

        cameras[frame_base_name] = {
            'intrinsics': K,
            'extrinsics': E
        }

    # write camera infos
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:
        pickle.dump(cameras, f)

    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:
        pickle.dump(mesh_infos, f)

    # write canonical joints
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    avg_betas_torch = torch.from_numpy(avg_betas[None, ...].copy())

    avg_global_t = np.mean(np.stack(all_global_t, axis=0), axis=0)
    avg_global_t_torch = torch.from_numpy(avg_global_t[None, None, ...].copy())

    mano = ManoLayer(use_pca=False, ncomps=45, side=hand_type,
                     flat_hand_mean=False, center_idx=None, mano_root=MODEL_DIR)

    # TODO-q: is it the right way to get canonical joints also for MANO, what are those?
    # _, template_joints = mano(torch.zeros(1, 48), avg_betas_torch, avg_global_t_torch)
    _, template_joints = mano(torch.zeros(1, 48), avg_betas_torch, global_t_1)
    # _, template_joints = mano(torch.zeros(1, 48), avg_betas_torch)
    template_joints_numpy = template_joints.detach().cpu().numpy()[0].copy()
    with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:
        pickle.dump(
            {
                'joints': template_joints_numpy,
            }, f)


if __name__ == '__main__':
    app.run(main)
