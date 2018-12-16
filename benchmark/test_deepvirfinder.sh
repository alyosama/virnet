genomes=("viral_test.fna" "non_viral_test.fna")
fragments=(500 1000 3000 100)
input_dir="../../data/3-fragments/fna/"
output_dir="../../benchmark/deep_results/"


for genome in ${genomes[@]}; do
    for fragment in ${fragments[@]}; do
        input_path=$input_dir$genome'_'$fragment'.fna'
        echo $input_path
        python ../../tools/DeepVirFinder/dvf.py -i $input_path -o $output_dir
    done
done
