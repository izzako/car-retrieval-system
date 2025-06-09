#!/bin/bash -l

# Set SCC project
#$ -P llamagrp

# Specify hard time limit for the job. 
#   The job will be aborted if it runs longer than this time.
#   The default time is 12 hours
#$ -l h_rt=48:00:00

# Send an email when the job finishes or if it is aborted (by default no email is sent).
#$ -m ea

# Give job a name
#$ -N object_det_IDD_40K_resnet50

# Combine output and error files into a single file
#$ -j y

# Request 2 CPU Core
#$ -pe omp 2

# Request 1 GPU 
#$ -l gpus=1

# Specify the minimum GPU compute capability. 
#$ -l gpu_c=8.0


# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "WORKING DIR: $TMPDIR"
echo "Job ID : $JOB_ID"
echo "=========================================================="


# load to the shared computing unit
module load cuda/12.2 gcc/12.2.0 python3/3.10.12 torch

# python -V

"/projectnb/llamagrp/izzan/env/bin/python" source_code/finetune_object_detection.py
