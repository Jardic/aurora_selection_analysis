#!/bin/bash

#SBATCH -J run_beam_params
#SBATCH --time=4:00:00
#SBATCH --partition=a36_any
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --mem=80G
#SBATCH --exclusive
#SBATCH --constraint=part-a
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jaroslav.kurfurst@uochb.cas.cz

./beam_search_params_effect_on_precision.py
