#!/usr/bin/bash
source activate tensorflow

fragments=(100 500 1000 3000)

for n in ${fragments[@]}; do
    echo "#prediction on $n bp files"
    python predict.py --input_dim=$n --input=../../data/3-fragments/fna/viral_test.fna_$n.fna --output=../../benchmark/vir_results/viral_test.fna_$n.fna --model_path=../../work_dir/models/saved_model/model_$n.h5
    python predict.py --input_dim=$n --input=../../data/3-fragments/fna/non_viral_test.fna_$n.fna --output=../../benchmark/vir_results/non_viral_test.fna_$n.fna --model_path=../../work_dir/models/saved_model/model_$n.h5
done

echo "#prediction of Virome"
python predict.py --input_dim=100 --input=../../data/4-metagenome/virome/virome-reads.fa --output=../../benchmark/vir_results/virome-reads.fa --model_path=../../work_dir/models/saved_model/model_100.h5

echo "#prediction of Microbiome"
python predict.py --input_dim=100 --input=../../data/4-metagenome/microbiome/microbiome-reads.fa --output=../../benchmark/vir_results/microbiome-reads.fa --model_path=../../work_dir/models/saved_model/model_100.h5
