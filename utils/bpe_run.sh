#BSUB -J BPE_TEST
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -W 99:99
#BSUB -o /home/alyabdel/workdir/test_%J.out
#BSUB -e /home/alyabdel/workdir/test_%J.err

num_operations=50000

echo "Strating for all genomes and num_operations=$num_operations"

echo "Cleaning Viral headers"
sed -e '/^>/ d'  ../../data/2-train_test/viral_train.fna > ../../work_space/viral_train.fna.txt
echo "Cleaning Non Viral headers"
sed -e '/^>/ d'  ../../data/2-train_test/non_viral_train.fna > ../../work_space/non_viral_train.fna.txt

echo "Combine viral and non viral files"
cat ../../work_space/viral_train.fna.txt ../../work_space/non_viral_train.fna.txt > ../../work_space/train.fna_all.txt

echo "Run BPE on the file"
../../subword-nmt/learn_bpe.py -s $num_operations < ../../work_space/train.fna_all.txt > ../../work_space/train_codes.fna_all.txt &


for N in 100 300 500 1000 3000; do
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
