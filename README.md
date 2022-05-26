# The Unsupervised Reinforcement Learning Benchmark (URLB)

URLB provides a set of leading algorithms for unsupervised reinforcement learning where agents first pre-train without access to extrinsic rewards and then are finetuned to downstream tasks.

This codebase was adapted from [DrQv2](https://github.com/facebookresearch/drqv2). The DDPG agent and training scripts were developed by Denis Yarats. All authors contributed to developing individual baselines for URLB.

## Requirements
We assume you have access to a GPU that can run CUDA 10.2 and CUDNN 8. Then, the simplest way to install all required dependencies is to create an anaconda environment by running
```sh
conda env create -f conda_env.yml
```
After the instalation ends you can activate your environment with
```sh
conda activate urlb
```

## Implemented Agents
| Agent | Command | Implementation Author(s) | Paper |
|---|---|---|---|
| ICM | `agent=icm` | Denis | [paper](https://arxiv.org/abs/1705.05363)|
| ProtoRL | `agent=proto` | Denis | [paper](https://arxiv.org/abs/2102.11271)|
| DIAYN | `agent=diayn` | Misha | [paper](https://arxiv.org/abs/1802.06070)|
| APT(ICM) | `agent=icm_apt` | Hao, Kimin | [paper](https://arxiv.org/abs/2103.04551)|
| APT(Ind) | `agent=ind_apt` | Hao, Kimin | [paper](https://arxiv.org/abs/2103.04551)|
| APS | `agent=aps` | Hao, Kimin | [paper](http://proceedings.mlr.press/v139/liu21b.html)|
| SMM | `agent=smm` | Albert | [paper](https://arxiv.org/abs/1906.05274) |
| RND | `agent=rnd` | Kevin | [paper](https://arxiv.org/abs/1810.12894) |
| Disagreement | `agent=disagreement` | Catherine | [paper](https://arxiv.org/abs/1906.04161) |

## Available Domains
We support the following domains.
| Domain | Tasks |
|---|---|
| `walker` | `stand`, `walk`, `run`, `flip` |
| `quadruped` | `walk`, `run`, `stand`, `jump` |
| `jaco` | `reach_top_left`, `reach_top_right`, `reach_bottom_left`, `reach_bottom_right` |


## Domain observation mode
Each domain supports two observation modes: states and pixels.
| Model | Command |
|---|---|
| states | `obs_type=states` |
| pixels | `obs_type=pixels` |


## Instructions
### Pre-training
To run pre-training use the `pretrain.py` script
```sh
python pretrain.py agent=icm domain=walker
```
or, if you want to train a skill-based agent, like DIAYN, run:
```sh
python pretrain.py agent=diayn domain=walker
```
This script will produce several agent snapshots after training for `100k`, `500k`, `1M`, and `2M` frames. The snapshots will be stored under the following directory:
```sh
./pretrained_models/<obs_type>/<domain>/<agent>/
```
For example:
```sh
./pretrained_models/states/walker/icm/
```

### Fine-tuning
Once you have pre-trained your method, you can use the saved snapshots to initialize the `DDPG` agent and fine-tune it on a downstream task. For example, let's say you have pre-trained `ICM`, you can fine-tune it on `walker_run` by running the following command:
```sh
python finetune.py pretrained_agent=icm task=walker_run snapshot_ts=1000000 obs_type=states
```
This will load a snapshot stored in `./pretrained_models/states/walker/icm/snapshot_1000000.pt`, initialize `DDPG` with it (both the actor and critic), and start training on `walker_run` using the extrinsic reward of the task.

For methods that use skills, include the agent, and the `reward_free` tag to false.
```sh
python finetune.py pretrained_agent=smm task=walker_run snapshot_ts=1000000 obs_type=states agent=smm reward_free=false
```

### Monitoring
Logs are stored in the `exp_local` folder. To launch tensorboard run:
```sh
tensorboard --logdir exp_local
```
The console output is also available in a form:
```
| train | F: 6000 | S: 3000 | E: 6 | L: 1000 | R: 5.5177 | FPS: 96.7586 | T: 0:00:42
```
a training entry decodes as
```
F  : total number of environment frames
S  : total number of agent steps
E  : total number of episodes
R  : episode return
FPS: training throughput (frames per second)
T  : total training time
```
