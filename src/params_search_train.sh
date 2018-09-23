#!/usr/bin/bash
source activate tensorflow

cell_types=("lstm" "gru" "rnn")
n_layers=(1 2 3)
nn=(128 256 512)

for cell_type in ${cell_types[@]}; do
	echo "Cell Type is $cell_type"
	for layer in ${n_layers[@]}; do
		echo "#Layers is $layer"
		for n in ${nn[@]}; do
			echo "#Neurons is $n"
			python train.py --cell_type=$cell_type --n_layers=$layer --n_neurons=$n
		done
	done
done