import os
import re
import gzip
import random
import time
import pandas as pd
import numpy as np
from Bio import SeqIO

#data_folder='E:/masters/virnet/data/1-genomes'
#output_folder='E:/masters/virnet/data/2-fragments'

data_folder='/media/aly/Work/masters/virnet/data/1-genomes'
output_folder='/media/aly/Work/masters/virnet/data/2-fragments/csv'

from dateutil import parser
import datetime
header_re = re.compile( r"ref\|(?P<ID>.*?)\|LEN=(?P<LEN>.*?)(\|DATE=(?P<DATE>.*?)){0,1}$", re.MULTILINE)
def extract_headerinfo(header):
    info=[m.groupdict() for m in header_re.finditer(header)]
    if info[0]['DATE'] is not None:
        info[0]['DATE']=parser.parse(info[0]['DATE'])
    info[0]['LEN']=int(info[0]['LEN'])
    return info[0]


def rand_parts(seq, n_gen, contigs_len):
    indices = range(len(seq) - (contigs_len - 1) * n_gen)
    result = []
    offset = 0
    for i in sorted(random.sample(indices, n_gen)):
        i += offset
        result.append(str(seq[i:i+contigs_len]))        
        offset += contigs_len - 1
    return result

def generate_seq(seq,contigs_len=500,is_random=False):
    if is_random:
       n_gen=random.randint(0,5)
       if (len(seq)>contigs_len):
           if (len(seq)<contigs_len*n_gen):
               start=random.randint(0,len(seq) - (contigs_len - 1))
               return [str(seq[start:start+contigs_len])]
           else:
               return rand_parts(seq,n_gen,contigs_len)
       else:
           return []
    else:
        seq_list=[]
        n_gen=len(seq)//contigs_len
        for i in range(n_gen):
            start=i*contigs_len   
            end=start+contigs_len
            seq_list.append(str(seq[start:end]))    
        ## Adding the last contigs with the perivous (The last contigs is overlapping TODO fix)
        seq_list.append(str(seq[-contigs_len:]))
        return seq_list

def load_data(file_path,n,sample=False):
    print('Processing {0} with n={1} and sample={2}'.format(file_path,n,sample))
    fragment_list=[]
    count=0
    for record in SeqIO.parse(file_path, "fasta"):
        count+=1
        generated_sequences=generate_seq(record.seq,contigs_len=n,is_random=sample)
        header=extract_headerinfo(record.id)
        for item in generated_sequences:
            fragment_list.append([header['ID'],header['LEN'],header['DATE'],item])
    print('generate from {0} genomes {1} fragments'.format(count,len(fragment_list)))
    return fragment_list

def generate_frament(file,n,sample=False):
    start=time.time()
    file_path=os.path.join(data_folder,file)
    fragment_list=load_data(file_path,n,sample)
    df=pd.DataFrame(fragment_list,columns=['ID','LEN','DATE','SEQ'])
    output_path=os.path.join(output_folder,"{0}_{1}.csv".format(file,n))
    print('Generated reads {0}'.format(len(df)))
    df.to_csv(output_path,index=False)
    end=time.time()
    print('Time elapased {0:3f} Secs'.format(end-start))


genome_files={'viral_train.fna':False,'viral_test.fna':False,'bacteria_train.fna':True,'bacteria_test.fna':True,'archaea_train.fna':True,'archaea_test.fna':True}
contigs_len=[100,150,300,500,1000,2000,3000,5000]
for file in genome_files:
    for clen in contigs_len:
        generate_frament(file,n=clen,sample=genome_files[file])




