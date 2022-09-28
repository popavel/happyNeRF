import glob
import json
import os.path

import yaml

from absl import app
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    '0000.yaml',
                    'the path of config file')

NUMBER_OF_FRAMES_TO_USE = 200


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

    pattern_shapes = os.path.join(path_shapes, '*.json')
    shapes_list = sorted(glob.glob(pattern_shapes))

    pattern_cameras = os.path.join(path_cameras, '*.json')
    cameras_list = sorted(glob.glob(pattern_cameras))

    assert len(shapes_list) == len(cameras_list), \
        f'shapes_list size: {len(shapes_list)} not equals cameras_list size: {len(cameras_list)}'

    shapes_and_cameras_list = list(zip(shapes_list, cameras_list))
    metadata = {}

    for i, (shape_file_path, camera_file_path) in enumerate(shapes_and_cameras_list):

        if i >= NUMBER_OF_FRAMES_TO_USE:
            break

        frame_id = os.path.basename(shape_file_path)[:-5]

        assert frame_id == os.path.basename(camera_file_path)[:-5], \
            f"frame ids for shapes and camera files must match. " \
            f"shape: {frame_id}, camera: {os.path.basename(camera_file_path)[:-5]}"

        with open(shape_file_path, 'r') as shape_file:
            shape_json = json.load(shape_file)

        with open(camera_file_path, 'r') as camera_file:
            camera_json = json.load(camera_file)

        frame_dictionary = {}

        frame_dictionary.update({'poses': shape_json['poses']})
        frame_dictionary.update({'betas': shape_json['shapes']})
        frame_dictionary.update({'global_t': shape_json['global_t']})

        frame_dictionary.update({'cam_intrinsics': camera_json['K'][0]})
        frame_dictionary.update({'cam_extrinsics': camera_json['M'][0]})

        metadata.update({frame_id: frame_dictionary})

    with open(os.path.join(path_output, 'metadata.json'), 'w') as output_file:
        json.dump(metadata, output_file)


if __name__ == '__main__':
    app.run(main)
