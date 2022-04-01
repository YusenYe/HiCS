# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:29:25 2021

@author: user
"""

#generate bin file from contact file
import os
import pandas as pd
import re

def extract_setname(filename,suffix):
    return re.sub("\.*"+suffix,"",os.path.basename(filename))

def get_proc_filenames(filenames, n_proc = 1, rank = 0):
    #print(rank, len(filenames))
    indices = list(range(rank, len(filenames), n_proc))
    proc_fnames = [filenames[i] for i in indices]
    return proc_fnames
    
def bin_file(filename,binsize,outdir,chr_columns,pos_columns,suffix,low_cutoff):
    name=extract_setname(filename,suffix)
    contact_statis=list()
    #if os.path.exists(os.path.join(os.path.join(outdir,name+'bedpe')))==0:  
        #filename=filenames[0]
    try:
        df=pd.read_csv(filename,sep='\t',header=0)
        #contact_statis.append(len(df))
        print(len(df))
        if len(df)>=20000:
            #chr_columns=[i-1 for i in chr_columns]
            #pos_columns=[i-1 for i in pos_columns]
            df=df.iloc[:,chr_columns+pos_columns]
            df.columns=["chr1","chr2","x1","y1"]
            df=df[abs(df['x1']-df['y1'])>=low_cutoff]
            contact_statis.append(len(df))
            df['x1']=(df['x1']//binsize*binsize).astype(int)
            df['y1']=(df['y1']//binsize*binsize).astype(int)
            #keep intra chr
            df=df[(df['chr1']==df['chr2'])]
            #keep autosomal
            df=df[df['chr1'].isin(['chr'+str(i) for i in range(23)])]
            #make x1 be the samller bin
            df.loc[df['x1']>df['y1'],['x1','y1']]=df.loc[df['x1']>df['y1'],['y1','x1']].values
            #remove duplicates rows
            df.drop_duplicates(inplace=True)
            #remove adjacent binpairs
            df=df[df['y1']-df['x1']>=binsize]
            df['x2']=(df['x1']+binsize).astype(int)
            df['y2']=(df['y1']+binsize).astype(int)
            df['count']=1
            contact_statis.append(len(df))
            df[['chr1','x1','x2','chr2','y1','y2','count']].to_csv(os.path.join(outdir,name+'.bedpe'),header=None,index=False)    
    except:
        print(filename)
        pass
    

    
def bin_sets(indir,suffix,binsize,outdir,chr_columns,pos_columns,low_cutoff,\
             n_proc = 1, rank = 0, logger = None):
    if logger:
        logger.set_rank(rank)
    if not outdir:
        outdir = indir
        outdir = os.path.join(outdir, "binned_data")
    
    try:
        os.makedirs(outdir)
    except:
        #print(e, 'excepting error in binning')
        pass
    
    #filenames = get_filepaths(indir, file_suffix)
    completed_filenames = os.listdir(outdir)
    completed_filenames=[name[:-len(".bedpe")] for name in completed_filenames]
    filenames=[os.path.join(indir,name) for name in os.listdir(indir) if ((name.endswith(suffix)) and (name[:-len(suffix)] \
                                                                                                     not in set(completed_filenames))) ]
    #filenames=[os.path.join(indir,name) for name in os.listdir(indir) if name.endswith(suffix)]
    proc_filenames = get_proc_filenames(filenames, n_proc, rank)
    
    contact_statis_all=pd.DataFrame()
    for k,filename in enumerate(proc_filenames):
        bin_file(filename,binsize,outdir,chr_columns,pos_columns,suffix,low_cutoff)
        #DF=pd.DataFrame(contact_statis).T
        #contact_statis_all=pd.concat([contact_statis_all,DF])
    return contact_statis_all

# using cooler plot
# import matplotlib.pyplot as plt
# filepath='/Users/yusen/YusenYe/Test/scHiC/out.cool'
# c = cooler.Cooler(filepath)

# c.matrix(balance=False, sparse=True)
# mat = c.matrix(balance=False, sparse=True)[1000:1200, 1000:1200]
# arr=mat.toarray()
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111)
# im = ax.matshow(np.log10(arr), cmap='YlOrRd')
# fig.colorbar(im)
    
    
    
    
    
    
    
    
    
