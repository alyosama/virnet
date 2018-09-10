import os
import re
import gzip
import random
import time
import pandas as pd
import numpy as np
from Bio import SeqIO
import sys
from pyfaidx import Fasta

#data_folder='E:/masters/virnet/data/1-genomes'
#output_folder='E:/masters/virnet/data/3-metagenome'

data_folder='/media/aly/Work/masters/virnet/data/1-genomes'
output_folder='/media/aly/Work/masters/virnet/data/3-metagenome'



random.seed(42)
def create_metagenome_genomes(file,reads,read_len,viral_ratio):
    print("Starting creating genomes for {0}".format(file))
    start=time.time()
    total_length=reads*read_len
    output_file=os.path.join(output_folder,"{0}.fna".format(file))


    viral_path=os.path.join(data_folder,'viral_test.fna')
    bacteria_path=os.path.join(data_folder,'bacteria_test.fna')
    archaea_path=os.path.join(data_folder,'archaea_test.fna')
    vfaa = list(Fasta(viral_path))
    bfaa = list(Fasta(bacteria_path))
    bfaa.extend(list(Fasta(archaea_path)))

    random.shuffle(vfaa)
    random.shuffle(bfaa)

    v_len=0
    v_count=0
    b_len=0
    b_count=0
    with open(output_file,'w+') as fout:
        print("Sample from Viruses")
        for sample in vfaa:
            if(v_len>=int(total_length*viral_ratio)):
                break
            fout.write('>{0}\n{1}\n'.format(sample.name,sample))
            v_len+=len(sample)
            v_count+=1
        print('Viruses Length {0} bp'.format(v_len))
        print('# Viruses Genomes {0}'.format(v_count))

        print("Sample from Bacteria")
        for sample in bfaa:
            if(v_len*1.0/(v_len+b_len)<=viral_ratio):
                break
            fout.write('>{0}\n{1}\n'.format(sample.name,sample))
            b_len+=len(sample)
            b_count+=1
    print('Bacteria Length {0} bp'.format(b_len))
    print('# Bacteria Genomes {0}'.format(b_count))
    end=time.time()
    print('Time elapased {0:3f} Secs'.format(end-start))
    print('Total Length {0} bp'.format(v_len+b_len))
    print('Bacteria ratio {0:.2f}%'.format(b_len*1.0/(v_len+b_len)*100))



create_metagenome_genomes('microbiome',reads=10**6,read_len=100,viral_ratio=0.25)
create_metagenome_genomes('virome',reads=10**6,read_len=100,viral_ratio=0.75)

