#!/bin/bash -l

# Please modify this file as needed, and upload it to your personal SCC folder (or a subdirectory thereof) at:
# /projectnb/lx496sh/students/your_bu_id

# To submit this job, cd into the folder where you uploaded this file and run:
# qsub run_bitfit_experiment.sh

# To change the session settings, see the official SCC instructions at:
# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/

#$ -P lx496sh       # This job is for LX 496/796
#$ -l h_rt=3:00:00  # Reserve the batch node for 3 hours
#$ -l gpus=1        # Request 1 GPU
#$ -l gpu_c=7.0     # Request a GPU that is v100 or better
#$ -N myjobname     # Give your job a name
#$ -m bea           # Send you an email when the job starts and ends
#$ -o log.txt       # Save anything printed to the terminal to a file...
#$ -j y             # ...including both print statements and error messages

module load miniconda
module load academic-ml/spring-2026
conda activate spring-2026-pyt
cd /projectnb/lx496sh/students/aaronep  # Modify this with your information
python train_model.py