# HappyNeRF: a model for learning appearance of human hands

This project is part of a master's thesis.

The code is based on humanNeRF implementation:
[Project Page](https://grail.cs.washington.edu/projects/humannerf/) |
[GitHub Repo](https://github.com/chungyiweng/humannerf)

## Prerequisites

### Configure conda environment

The necessary dependencies can be installed with the provided conda environment.yml file.

    conda env create -f environment.yml --prefix /path_to_environment/happynerf_env

Additionally, pyTorch 1.8.1 and chumpy are required.

    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

### Download MANO model

Download the MANO model, files MANO_LEFT.pkl and MANO_RIGHT.pkl from [here](https://mano.is.tue.mpg.de/index.html),
and put them in `third_parties/mano/manopth_hanco/manopth/models_for_manopth`.

## Running the code

For additional information on running the code, check the 
[humanNeRF repository](https://github.com/chungyiweng/humannerf).

Some already preprocessed data can be found in the folder `dataset`.
To use them for running the code, name the corresponding folder `monocular`
and put it in the folder `dataset/wild`.

PyCharm users can find run configuration files in the folder `PyCharm_run_configurations`
Otherwise, following commands can be used as a reference.

### Train a model

#### Multiple GPUs 

    python train.py --cfg configs/human_nerf/wild/monocular/adventure.yaml

#### Single GPU

    python train.py --cfg configs/human_nerf/wild/monocular/single_gpu.yaml

### Render output

Render the frame input (i.e., observed motion sequence).

    python run.py --type movement --cfg configs/human_nerf/wild/monocular/single_gpu.yaml 

Run free-viewpoint rendering on a particular frame (e.g., frame 128).

    python run.py --type freeview --cfg configs/human_nerf/wild/monocular/adventure.yaml freeview.frame_idx 8

## Evaluation

The evaluation script with some already prepared data can be found in the folder `evaluation`.

You can either use the provided Run Configurations or use the following command.
    
    python evaluate.py --cfg evaluate.yaml    

Change the contents of the `evaluate.yaml` file depending on the data you want to use.

## Preprocessing

Preprocessing scripts can be found in the folder `tools`.
First, you need to prepare the metadata. 
The corresponding script is located in the folder `tools/prepare_[data set name]/prepare_metadata`.
Put the generated `metadata.json` file to the folder `dataset/[name of the dataset]/monocular`.
Second, run the preprocessing script located in the folder `tools/prepare_[data set name]`.
The generated files are placed in the folder `dataset/[name of the dataset]/monocular`.

You can either use the provided Run Configurations 
or use the following commands as an example.

    python prepare_metadata_json.py --cfg 0000.yaml
    python prepare_dataset.py --cfg hanco.yaml
    

For additional information on the format of the `metadata.json` file refer to
the `README.md` file from [humanNeRF repository](https://github.com/chungyiweng/humannerf).

You can generate masks for interHand2.6M dataset by cloning the
[interHand2.6M repository](https://github.com/facebookresearch/InterHand2.6M#mano-mesh-rendering-demo)
and running `render.py` script in `tool/MANO_render`. 
To generate masks you need a modified version of `render.py` script, 
which we provide in the folder `tools/get_masks_for_interhand`.