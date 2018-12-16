#!/usr/bin/bash
source activate tensorflow

cell_types=("lstm" "gru" "rnn")
n_layers=(1 2 3)
embed_sizes=(64 128 256)
ngrams=(3 5 7)

for cell_type in ${cell_types[@]}; do
	for layer in ${n_layers[@]}; do
		for n in ${embed_sizes[@]}; do
			for ngram in  ${ngrams[@]}; do
				echo "###################################################################"
				echo "Cell Type is $cell_type"
				echo "#Layers is $layer"
				echo "#Embedded size is $n"
				echo "#ngram is $ngram"
				python train.py --sample 250000 --batch_size=256 --epoch=10 --cell_type=$cell_type --n_layers=$layer --embed_size=$n --ngrams=$ngram
			done
		done
	done
done