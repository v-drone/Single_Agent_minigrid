#!/bin/bash
module load python/anaconda3
source $condaDotFile
source activate gymnasium

nvidia-smi

which python

# Define paths
_path="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/Single_Agent_minigrid/"
_log="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/logging/drone/RouteWithTrod/"
_checkpoint="/jmain02/home/J2AD006/jxb06/cxz46-jxb06/checkpoints/drone/RouteWithTrod/"

# Change directory
cd $_path || exit

# Run Python script
python $_path/apex_dpber_jade1.py \
    -R $SLURM_JOB_ID \
    -S $_path/apex.yml \
    -L $_log \
    -C $_checkpoint \
    -E RouteWithTrod
