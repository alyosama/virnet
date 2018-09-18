#! /usr/bin/env bash

#BSUB -J synergy_test_v
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -W 99:99
#BSUB -o /home/alyabdel/workdir/test_%J.out
#BSUB -e /home/alyabdel/workdir/test_%J.err

Rscript /gpfs/home/alyabdel/local/virnet/benchmark/test_virfinder.R