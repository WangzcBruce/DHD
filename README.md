
# Drones Help Drones: A Collaborative Framework for Multi-Drone Object Trajectory Prediction and Beyond

Welcome to the official PyTorch implementation of **"Drones Help Drones: A Collaborative Framework for Multi-Drone Object Trajectory Prediction and Beyond."** We have open-sourced this repository to foster research and collaboration in the field of multi-drone trajectory prediction and related areas.

## Code Availability

The implementation code is now accessible for public use and contribution.

### Latest News
**"Drones Help Drones"** has been accepted for presentation at the **Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024).**

## Setup Instructions

### Step 1: Create the Conda Environment

To set up the environment, use the following command:

```bash
conda env create -f environment.yml
```
### Step 2: Replace `splits.py`

Ensure you replace the `splits.py` file in the `nuscenes` package (typically found at `/miniconda3/envs/dhd/lib/python3.7/site-packages/nuscenes/utils/splits.py`) with our provided version of `splits.py`.

### Step 3: Download the Dataset

Download the complete **Air-Co-Pred** dataset, which includes the Trainval dataset (metadata and file blobs parts 0-36), from the following link:

[Download Link](https://pan.baidu.com/s/1XRgtXcLHS4fk02EqE-mYUQ)  
Access Code: `4av8`

Once downloaded, extract the `.tar` files into your desired data root directory (`YOUR_DATAROOT`), organizing them as follows:

```bash
Air-Co-Pred/
├── trainval/
│   ├── maps/
│   ├── samples/
│   ├── sweeps/
│   └── v1.0-trainval/
```

## Model Training

To train the DHD (Drones Help Drones) model, execute the following command:

```bash
python train.py --config=dhd/config/dhd.yml \
                LOG_DIR xxx \
                GPUS [x,x,x,x] \
                BATCHSIZE 1 \
                DATASET.DATAROOT YOUR_DATAROOT
```

## Model Evaluation

To evaluate the model with pre-trained weights, run:

```bash
python test.py --config dhd/config/dhd.yml \
                PRETRAINED.LOAD_WEIGHTS True \
                PRETRAINED.PATH $YOUR_PRETRAINED_WEIGHTS_PATH \
                GPUS [x,x,x,x] \
                BATCHSIZE 1 \
                DATASET.DATAROOT YOUR_DATAROOT
```

## Citation

If you find this work helpful in your research, please consider citing us:

```bibtex
@inproceedings{
  title={Drones Help Drones: A Collaborative Framework for Multi-Drone Object Trajectory Prediction and Beyond},
  author={Wang Z, Cheng P, Chen M, Tian P, Wang Z, Li X, Yang X, Sun X.},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```
