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

## License

The first-party HappyNeRF code in this repository is released under the MIT
License — see [LICENSE](LICENSE). This project is derived from
[humanNeRF](https://github.com/chungyiweng/humannerf), which is also MIT-licensed;
its copyright notice is retained in [LICENSE](LICENSE).

The MIT grant does **not** cover the bundled third-party code under
`third_parties/` or the preprocessed data under `dataset/`, which keep their own
licenses. In particular, `third_parties/mano/manopth_hanco/manopth/` bundles two
separately-licensed components: the manopth differentiable MANO layer
(`manolayer.py`, `rodrigues_layer.py`, `rot6d.py`, and supporting files) is
GPL-3.0, while the MANO/SMPL+H model code in the same directory
(`load_util.py`, `posemapper.py`) is restricted to non-commercial research use.
The MANO/SMPL+H model is also used outside that directory by
`third_parties/mano/mano_numpy.py`, which is likewise limited to non-commercial
research use.
The InterHand2.6M-derived files (`tools/get_masks_for_interhand/render.py`,
`dataset/interhand/`) and the HanCo data (`dataset/hanco/`) are likewise
restricted to non-commercial research use. The bundled monocular example data
under `dataset/wild/` consists of preprocessed body fits produced with the SMPL
body model (Max Planck Gesellschaft) and derived from the humanNeRF example, so
it carries the SMPL non-commercial research-only restriction together with the
terms of its upstream image source. See
[the manopth NOTICE](third_parties/mano/manopth_hanco/NOTICE), the per-file
headers, and the upstream MANO / SMPL / InterHand2.6M / HanCo licenses for the
exact terms.

The other bundled third-party libraries under `third_parties/` are permissively
licensed but still carry their own terms: `third_parties/lpips/` is BSD-2-Clause
(© 2018 Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver
Wang) and `third_parties/yacs/` is Apache-2.0 with a NOTICE. Redistributions
must retain their copyright notices, and — per Apache-2.0 §4(d) — must carry the
contents of [`third_parties/yacs/NOTICE`](third_parties/yacs/NOTICE) (see also
[`third_parties/yacs/LICENSE`](third_parties/yacs/LICENSE)).

Note that two otherwise-MIT first-party scripts —
`tools/prepare_hanco/prepare_dataset.py` and
`tools/prepare_interhand/prepare_dataset.py` — import the GPL-3.0 manopth layer
(`ManoLayer`). That layer in turn imports the non-commercial MANO/SMPL+H model
code (`load_util.py`, `posemapper.py`) and loads the non-commercial MANO model
files at runtime. So when these scripts are run or redistributed together with
the manopth layer they form a combined work bound by **both** the GPL-3.0 terms
*and* the MANO/SMPL+H non-commercial research-only restriction — the MIT terms
alone do not cover that combination, and the non-commercial restriction means it
may not be used commercially even when the GPL-3.0 obligations are met.
