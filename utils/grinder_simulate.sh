#!/bin/bash         

echo "Cleaning file"
#sed -e '/^[^>]/s/[^ATGCatgc]/N/g' microbiome.fna > microbiome_clean.fna

echo "Start Grinder"
#grinder -reference_file microbiome_clean.fna -mutation_dist poly4 3e-3 3.3e-8 -mutation_ratio 91 9 -total_reads 1000000 -base_name microbiome -output_dir microbiome

echo "Cleaning file"
#sed -e '/^[^>]/s/[^ATGCatgc]/N/g' virome.fna > virome_clean.fna

echo "Start Grinder"
#grinder -reference_file virome_clean.fna -mutation_dist poly4 3e-3 3.3e-8 -mutation_ratio 91 9 -total_reads 1000000 -base_name virome -output_dir virome



echo "Start Grinder for 500 bp"
grinder -reference_file microbiome_clean.fna -mutation_dist poly4 3e-3 3.3e-8 -mutation_ratio 91 9 -total_reads 1000000 -base_name microbiome_500 -output_dir microbiome -rd 500


echo "Start Grinder for 500 bp"
grinder -reference_file virome_clean.fna -mutation_dist poly4 3e-3 3.3e-8 -mutation_ratio 91 9 -total_reads 1000000 -base_name virome_500 -output_dir virome -rd 500
