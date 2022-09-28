import glob
import json
import os.path

import yaml

from absl import app
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    'Capture0.yaml',
                    'the path of config file')

NUMBER_OF_FRAMES_TO_USE = 1000

# whether to rename input frames to the format [camera idx]_[frame idx]
# enable if not yet renamed
RENAME = False


def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)
    return config


def main(argv):
    del argv  # unused

    cfg = parse_config()

    path_shapes = cfg['path_shapes']
    path_cameras = cfg['path_cameras']
    path_output = cfg['path_output']

    split = cfg['split']
    capture_idx = cfg['capture_idx']
    cam_indices = cfg['cam_indices']
    hand_type = cfg['hand_type']

    paths_img = cfg['paths_img']

    imgs_and_cams = list(zip(paths_img, cam_indices))

    with open(os.path.join(path_shapes, 'InterHand2.6M_' + split + '_MANO_NeuralAnnot.json')) as f:
        mano_params = json.load(f)

    with open(os.path.join(path_cameras, 'InterHand2.6M_' + split + '_camera.json')) as f:
        cam_params = json.load(f)

    metadata = {}
    for (path_img, cam_idx) in imgs_and_cams:

        img_path_list = sorted(glob.glob(os.path.join(path_img, '*.jpg')))

        for i, img_path in enumerate(img_path_list):

            if i >= NUMBER_OF_FRAMES_TO_USE:
                break

            frame_dictionary = {}

            frame_id = img_path.split('/')[-1][:-4]

            if RENAME:
                frame_idx = frame_id[5:]
            else:
                frame_idx = frame_id[7:]

            try:
                mano_param = mano_params[capture_idx][frame_idx][hand_type]
                if mano_param is None:
                    raise Exception("No values for given keys")
            except KeyError:
                raise Exception("No such keys")

            poses = {'poses': [
                mano_param['pose']
            ]}

            betas = {'betas': [
                mano_param['shape']
            ]}

            global_t = {'global_t': [
                [
                    mano_param['trans']
                ]
            ]}

            frame_dictionary.update(poses)
            frame_dictionary.update(betas)
            frame_dictionary.update(global_t)

            cam_param = cam_params[capture_idx]

            t = cam_param['campos'][cam_idx]
            R = cam_param['camrot'][cam_idx]

            # Rt = {'cam_extrinsics': [
            #     [R[0][0], R[0][1], R[0][2], t[0]],
            #     [R[1][0], R[1][1], R[1][2], t[1]],
            #     [R[2][0], R[2][1], R[2][2], t[2]],
            #     [0.0, 0.0, 0.0, 1.0]
            # ]}

            # Rt = {'cam_extrinsics': [
            #     [R[0][0], R[0][1], R[0][2], t[0] / 1000.0],
            #     [R[1][0], R[1][1], R[1][2], t[1] / 1000.0],
            #     [R[2][0], R[2][1], R[2][2], t[2] / 1000.0],
            #     [0.0, 0.0, 0.0, 1.0]
            # ]}

            # Rt = {'cam_extrinsics': [
            #     [R[0][0], R[1][0], R[2][0], t[0]],
            #     [R[0][1], R[1][1], R[2][1], t[1]],
            #     [R[0][2], R[1][2], R[2][2], t[2]],
            #     [0.0, 0.0, 0.0, 1.0]

            Rt = {'cam_extrinsics': [
                [R[0][0], R[1][0], R[2][0], t[0] / 1000.0],
                [R[0][1], R[1][1], R[2][1], t[1] / 1000.0],
                [R[0][2], R[1][2], R[2][2], t[2] / 1000.0],
                [0.0, 0.0, 0.0, 1.0]
            ]}

            f = cam_param['focal'][cam_idx]
            c = cam_param['princpt'][cam_idx]

            K = {'cam_intrinsics': [
                [f[0], 0.0, c[0]],
                [0.0, f[1], c[1]],
                [0.0, 0.0, 1.0]
            ]}

            frame_dictionary.update(K)
            frame_dictionary.update(Rt)

            new_frame_id = cam_idx + '_' + frame_idx

            if RENAME:
                src = os.path.join(path_img, frame_id + '.jpg')
                dst = os.path.join(path_img, new_frame_id + '.jpg')
                os.rename(src, dst)

            metadata.update({new_frame_id: frame_dictionary})

    with open(os.path.join(path_output, 'metadata.json'), 'w') as output_file:
        json.dump(metadata, output_file)


if __name__ == '__main__':
    app.run(main)
