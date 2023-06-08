#!/bin/bash

#SBATCH --partition=psanaq

source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh
conda activate deeplearning8
base_path=/cds/home/i/isele/tmox51020/results/erik/nn_reconst # changeme - where is the preproc.py file
script=$base_path/trn_gpu_fc.py # changeme
log=/cds/home/i/isele/tmox51020/results/erik/nn_reconst/logs/$1  

if [ -z "$4" ]
then
    n_nodes=1
else
    n_nodes=$2
fi

if [ -z "$5" ]
then
    tasks_per_node=1 #I think there are 16 procs per node for the feh queues, 64 for the ffb and 12 for the 'old' psana qs
else
    tasks_per_node=$3
fi

echo $log
echo $script

sbatch -p psanagpuq  --gpus-per-node=1 -N $n_nodes --ntasks-per-node $tasks_per_node --output $log --wrap="srun python -u $script $1"

#

