import pandas as pd

import os
data_folder='/media/aly/Work/masters/virnet/data/2-fragments'

def csv_to_fasta(file):
    file_path=os.path.join(data_folder,'csv',file)
    df=pd.read_csv(file_path)

    output_path=os.path.join(data_folder,'fna',file.replace(".csv",".fna"))
    with open(output_path,'w+') as fin:
        for index, row in df.iterrows():
            fin.write('>{0} {1}\n{2}\n'.format(row['ID'],index,row['SEQ']))

for file in os.listdir(os.path.join(data_folder,'csv')):
    if 'test' in file:
        print('Converting {0}'.format(file))
        csv_to_fasta(file)
