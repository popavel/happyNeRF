# HappyNeRF: a model for learning appearance of human hands

This project is part of a master's thesis.

The code is based on humanNeRF implementation:
[Project Page](https://grail.cs.washington.edu/projects/humannerf/) |
[GitHub Repo](https://github.com/chungyiweng/humannerf)

## Prerequisite

### `Configure environment`

The necessary dependencies can be installed with the provided conda environment.yml file.

    conda env create -f environment.yml --prefix /path_to_environment/happynerf_env

Additionally, pyTorch 1.8.1 is required.

    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

### `Download MANO model`

Download the MANO model, files MANO_LEFT.pkl and MANO_RIGHT.pkl from [here](https://mano.is.tue.mpg.de/index.html),
and put them in `third_parties/mano/manopth_hanco/manopth/models_for_manopth`.

### `Running the code`

PyCharm users can find run configuration files in the folder `PyCharm_run_configurations`