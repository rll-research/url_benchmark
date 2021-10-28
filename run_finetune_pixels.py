#!/usr/bin/env python3
import time
import os
import sys
import argparse
import pathlib, shutil
from datetime import datetime
from subprocess import Popen, DEVNULL

from dmc_benchmark import TASKS


class Overrides(object):
    def __init__(self):
        self.kvs = dict()

    def add(self, key, value):
        if type(value) == list:
            value = ','.join(str(v) for v in value)
        else:
            value = str(value)
        assert key not in self.kvs
        self.kvs[key] = value

    def cmd(self):
        cmd = []
        for k, v in self.kvs.items():
            cmd.append(f'{k}={v}')
        return cmd


def make_code_snap(experiment, slurm_dir='exp_sweep'):
    now = datetime.now()
    snap_dir = pathlib.Path.cwd() / slurm_dir
    snap_dir /= now.strftime('%Y.%m.%d')
    snap_dir /= now.strftime('%H%M%S') + f'_{experiment}'
    snap_dir.mkdir(exist_ok=True, parents=True)

    def copy_dir(dir, pat):
        dst_dir = snap_dir / 'code' / dir
        dst_dir.mkdir(exist_ok=True, parents=True)
        for f in (src_dir / dir).glob(pat):
            shutil.copy(f, dst_dir / f.name)

    dirs_to_copy = ['.', 'agent', 'custom_dmc_tasks']
    src_dir = pathlib.Path.cwd()
    for dir in dirs_to_copy:
        copy_dir(dir, '*.py')
        copy_dir(dir, '*.yaml')

    return snap_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('agent', type=str)
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()

    experiment = f'finetune_pixels_{args.agent}'
    snap_dir = make_code_snap(experiment)
    print(str(snap_dir))

    overrides = Overrides()

    # slurm params
    overrides.add('hydra/launcher', 'submitit_slurm')
    overrides.add('hydra.launcher.partition', 'learnfair')
    overrides.add('hydra.sweep.dir', str(snap_dir))
    overrides.add('hydra.launcher.submitit_folder', str(snap_dir / 'slurm'))
    overrides.add('experiment', experiment)

    # agent
    overrides.add('agent', args.agent)
    # env
    overrides.add('task', TASKS)

    # states specific params
    overrides.add('obs_type', 'pixels')
    overrides.add('action_repeat', 2)
    overrides.add('frame_stack', 3)
    overrides.add('agent.batch_size', 256)

    # train params
    overrides.add('num_train_frames', 500010)
    # eval params
    overrides.add('eval_every_frames', 10000)
    overrides.add('num_eval_episodes', 10)

    # sweep params
    overrides.add('snapshot_base_dir', str(pathlib.Path.cwd() / 'rerun_models'))
    overrides.add('snapshot_ts', [100000, 500000, 1000000, 2000000])
    # seeds
    overrides.add('seed', list(range(1, 11)))

    cmd = ['python', str(snap_dir / 'code' / 'finetune.py'), '-m']
    cmd += overrides.cmd()

    if args.dry:
        print(' '.join(cmd))
    else:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(snap_dir / 'code')
        p = Popen(cmd, env=env)
        p.communicate()


if __name__ == '__main__':
    main()
