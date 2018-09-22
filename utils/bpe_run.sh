#BSUB -J BPE_TEST
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -W 99:99
#BSUB -o /home/alyabdel/workdir/test_%J.out
#BSUB -e /home/alyabdel/workdir/test_%J.err

num_operations=50000
for N in 100 300 500 1000 5000; do
    echo "Strating N=$N and num_operations=$num_operations"

    echo "Cleaning non_viral headers"
    sed -e '/^>/ d'  ../../data/3-fragments/fna/non_viral_train.fna_$N.fna > ../../work_space/non_viral_train.fna_$N.txt
    echo "Cleaning viral headers"
    sed -e '/^>/ d'  ../../data/3-fragments/fna/viral_train.fna_$N.fna > ../../work_space/viral_train.fna_$N.txt

    echo "Combine viral and non viral files"
    cat ../../work_space/non_viral_train.fna_$N.txt ../../work_space/viral_train.fna_$N.txt > ../../work_space/train.fna_$N.txt

    echo "Run BPE on the file"
    ../../subword-nmt/learn_bpe.py -s $num_operations < ../../work_space/train.fna_$N.txt > ../../work_space/train_codes.fna_$N.txt &
done 