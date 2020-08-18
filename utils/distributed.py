"""
Utilities for distributed training.

This is setup to work at NERSC, and just a few things are supported:
    - MPI backend
    - NCCL backend with ranks determined by SLURM variables and intialized via
      shared file under $SCRATCH.
    - GLOO backend with rank determined by SLURM variables and intialized via
      shared file under $SCRATCH.
"""

import os
import torch.distributed as dist

def _get_sync_file():
    """Logic for naming sync file using slurm env variables"""
    sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH']
    os.makedirs(sync_file_dir, exist_ok=True)
    sync_file = 'file://%s/pytorch_sync.%s.%s' % (
        sync_file_dir, os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
    return sync_file

def init_workers_gloo_file():
    """Initialize workers with GLOO backend and sync file"""
    rank = int(os.environ['SLURM_PROCID'])
    n_ranks = int(os.environ['SLURM_NTASKS'])
    sync_file = _get_sync_file()
    dist.init_process_group(backend='gloo', world_size=n_ranks, rank=rank,
                            init_method=sync_file)
    return rank, n_ranks

def init_workers_nccl_file():
    """Initialize workers with NCCL backend and sync file"""
    rank = int(os.environ['SLURM_PROCID'])
    n_ranks = int(os.environ['SLURM_NTASKS'])
    sync_file = _get_sync_file()
    print('Setting up with sync file', sync_file)
    dist.init_process_group(backend='nccl', world_size=n_ranks, rank=rank,
                            init_method=sync_file)
    return rank, n_ranks

def init_workers_nccl_lsf():
    """Initialize workers with NCCL backend and LSF environment"""
    import subprocess
    get_master = "echo $(cat {} | sort | uniq | grep -v batch | grep -v login | head -1)".format(
        os.environ['LSB_DJOB_HOSTFILE'])
    os.environ['MASTER_ADDR'] = str(subprocess.check_output(get_master, shell=True))[2:-3]
    os.environ['MASTER_PORT'] = "23456"
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    rank = int(os.environ['RANK'])
    n_ranks = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='nccl', world_size=n_ranks, rank=rank)
    return rank, n_ranks

def init_workers_mpi():
    """Initialize workers with MPI backend"""
    dist.init_process_group(backend='mpi')
    rank = dist.get_rank()
    n_ranks = dist.get_world_size()
    return rank, n_ranks

def init_workers(backend=None):
    """Initialize workers for specified backend.

    Note that only a few modes are currently supported:
    - MPI backend
    - NCCL backend with ranks determined by SLURM variables and intialized via
      shared file under $SCRATCH.
    - GLOO backend with rank determined by SLURM variables and intialized via
      shared file under $SCRATCH.
    """
    if backend is None:
        rank, n_ranks = 0, 1
    elif backend == 'mpi':
        rank, n_ranks = init_workers_mpi()
    elif backend == 'nccl':
        rank, n_ranks = init_workers_nccl_file()
    elif backend == 'nccl-lsf':
        rank, n_ranks = init_workers_nccl_lsf()
    elif backend == 'gloo':
        rank, n_ranks = init_workers_gloo_file()
    return rank, n_ranks

def try_barrier():
    """Attempt a barrier but ignore any exceptions"""
    try:
        dist.barrier()
    except:
        pass
