import os
import re
import gzip
import random
import time
import pandas as pd
import numpy as np
from Bio import SeqIO

data_folder='E:/refseq/'
output_folder='E:/masters/virnet/data/1-genomes'




##################################### Viruses #############################################
random.seed(42)
def load_gbfile(gb_file,sample):
    genomes=[]
    if(sample==True and random.randint(0,2)!=0):
        return genomes
    with gzip.open(gb_file,"rt") as handle:
        for record in SeqIO.parse(handle, "genbank"):
            if(sample==True and random.randint(0,11)!=0):
                continue
            else:
                genomes.append([record.id,len(record),record.annotations['date'],record.seq])   
    return genomes  

def load_data(genometype,sample=False):
    start=time.time()
    genomes=[]
    genome_folder=os.path.join(data_folder,genometype)
    files_list=[file for file in os.listdir(genome_folder) if file.endswith(".genomic.gbff.gz")]
    count=1
    for file in files_list:
        print('{1:.2f}% Prasing {0} file'.format(file,count*1.0/len(files_list)*100))
        gb_file=os.path.join(genome_folder,file)
        file_genomes=load_gbfile(gb_file,sample)
        genomes.extend(file_genomes)
        count+=1
            
    print('Saving {0} genomes'.format(len(genomes)))
    file_name=os.path.join(output_folder,"{0}.fna".format(genometype))
    with open(file_name,'w+') as f:
        for genome in genomes:
            f.write('>ref|{0}|LEN={1}|DATE={2}\n{3}\n'.format(genome[0],genome[1],genome[2],genome[3]))
    end=time.time()
    print('Time elapased {0:3f} Sec'.format(end-start))

import re
from dateutil import parser
import datetime
viral_threshold=datetime.datetime(2017,1,1)
header_re = re.compile( r"ref\|(?P<ID>.*?)\|LEN=(?P<LEN>.*?)\|DATE=(?P<DATE>.*?)$", re.MULTILINE)
def extract_headerinfo(header):
    info=[m.groupdict() for m in header_re.finditer(header)]
    info[0]['DATE']=parser.parse(info[0]['DATE'])
    info[0]['LEN']=int(info[0]['LEN'])
    return info[0]

def split_genomes(genometype):
    start=time.time()
    fasta_file=os.path.join(output_folder,"{0}.fna".format(genometype))
    train_file=os.path.join(output_folder,"{0}_train.fna".format(genometype))
    test_file=os.path.join(output_folder,"{0}_test.fna".format(genometype))
    count=0
    t_count=0
    print('Reading and Splitting {0} genomes'.format(genometype))
    with open(fasta_file,"rt") as handle , open(train_file,'w+') as train_f, open(test_file,'w+') as test_f:
        for record in SeqIO.parse(handle, "fasta"):
            header=extract_headerinfo(record.id)
            if(header['DATE']<viral_threshold):
                train_f.write('>{0}\n{1}\n'.format(record.id,record.seq))
                count+=1
            else:
                test_f.write('>{0}\n{1}\n'.format(record.id,record.seq))
                t_count+=1
    end=time.time()
    print('Training {0} genomes'.format(count))
    print('Testing {0} genomes'.format(t_count))
    print('Time elapased {0:3f} Sec'.format(end-start))

load_data('viral',sample=False)
split_genomes('viral')



##################################### BACTERIA #############################################
random.seed(42)
def load_fasta_file(gb_file,sample):
    genomes=[]
    if(sample==True and random.randint(0,2)!=0):
        return genomes
    with gzip.open(gb_file,"rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            if(sample==True and random.randint(0,11)!=0):
                continue
            else:
                genomes.append([record.id,len(record),record.seq])   
    return genomes  

def load_fasta_data(genometype,sample=False):
    start=time.time()
    genomes=[]
    genome_folder=os.path.join(data_folder,genometype)
    files_list=[file for file in os.listdir(genome_folder) if file.endswith(".genomic.fna.gz")]
    count=1
    for file in files_list:
        print('{1:.2f}% Prasing {0} file'.format(file,count*1.0/len(files_list)*100))
        gb_file=os.path.join(genome_folder,file)
        file_genomes=load_fasta_file(gb_file,sample)
        genomes.extend(file_genomes)
        count+=1
            
    print('Saving {0} genomes'.format(len(genomes)))
    file_name=os.path.join(output_folder,"{0}.fna".format(genometype))
    with open(file_name,'w+') as f:
        for genome in genomes:
            f.write('>ref|{0}|LEN={1}\n{2}\n'.format(genome[0],genome[1],genome[2]))
    end=time.time()
    print('Time elapased {0:3f} Sec'.format(end-start))

def split_sample_genomes(genometype):
    start=time.time()
    fasta_file=os.path.join(output_folder,"{0}.fna".format(genometype))
    train_file=os.path.join(output_folder,"{0}_train.fna".format(genometype))
    test_file=os.path.join(output_folder,"{0}_test.fna".format(genometype))
    count=0
    t_count=0
    print('Reading and Splitting {0} genomes'.format(genometype))
    with open(fasta_file,"rt") as handle , open(train_file,'w+') as train_f, open(test_file,'w+') as test_f:
        for record in SeqIO.parse(handle, "fasta"):
            toss=random.randint(1,10)
            if(toss==4 or toss==7):
                test_f.write('>{0}\n{1}\n'.format(record.id,record.seq))
                t_count+=1
            else:
                train_f.write('>{0}\n{1}\n'.format(record.id,record.seq))
                count+=1
    end=time.time()
    print('Training {0} genomes'.format(count))
    print('Testing {0} genomes'.format(t_count))
    print('Time elapased {0:3f} Sec'.format(end-start))

load_fasta_data('archaea',sample=True)
split_sample_genomes('archaea')

load_fasta_data('bacteria',sample=True)
split_sample_genomes('bacteria')
