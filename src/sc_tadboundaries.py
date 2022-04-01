# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 18:23:05 2021

@author: user
"""
import os
from scipy import sparse
import numpy as np
import pandas as pd

def get_proc_chroms(chrom_lens, rank, n_proc):
    chrom_list = [(k, chrom_lens[k]) for k in list(chrom_lens.keys())]
    chrom_list.sort(key = lambda x: x[1])
    chrom_list.reverse()
    chrom_names = [i[0] for i in chrom_list]
    #chrom_names = list(chrom_lens.keys())
    #chrom_names.sort()
    
    indices = list(range(rank, len(chrom_names), n_proc))
    proc_chroms = [chrom_names[i] for i in indices]
    return proc_chroms

import src.sctad_boundaries as sb
def Detecting_scTAD(mat_3d,window=20,scale=1):
    
    ins_strengthes=sb.compute_boundaries(mat_3d,window)
    ins_strengthes_chr=ins_strengthes.to_numpy()
    #detecting　candidated TAD boundaries of each single cells
    cluster_results=pd.DataFrame()
    num_cells=ins_strengthes.shape[0]
    boundaries_records=[]
    
    for i in range(num_cells):
        #print(i)
        vector=ins_strengthes_chr[i,:]
        #vector=ins_vector[tad_limit_upper:(ins_vector.shape[0]-tad_limit_upper)]
        hdmd_vector=sb.detect_can_boundaries(vector,max_dis=500)
        boundaries_record=pd.DataFrame()
        boundaries_record['den']=vector
        boundaries_record['hdmd']=hdmd_vector
        boundaries_record=sb.cluster_result(boundaries_record,scale)
        
        gap=vector.copy()
        gap[gap!=1]=0
        gap[gap==1]=np.nan
        cluster_results[i]=boundaries_record['cluster']+gap
        boundaries_records.append(boundaries_record)
        #print(chrom,i,':',cluster_results.shape[0])
        #boundaries_record['cluster']=boundaries_record['cluster']+gap
        #filtering boundaries using file
    return cluster_results,boundaries_records,ins_strengthes_chr

def sctad_boundary(indir,outdir, contact_statis_pd ,chrom_lens, binsize, dist, tad_limit_upper, \
                      rank = 0, n_proc = 1, max_mem = 2, logger = None, window=20,scale=1):
    """
    detecting TAD boundaries from single cells

    """
    import time
    logger.set_rank(rank)
    try:
        os.makedirs(outdir)
    except:
        pass
    proc_chroms = get_proc_chroms(chrom_lens, rank, n_proc)
    completed_filenames = os.listdir(indir)
    filter_cells=contact_statis_pd.loc[contact_statis_pd[1]>250000,0].tolist()
    
    for chrom in proc_chroms:
        t=time.time()
        #chrom=proc_chroms[0]
        if ~os.path.exists(os.path.join(outdir, ".".join(["ins_strengthes", chrom, "npy"]))):
            using_suffix=".".join([chrom,  "rl", "npz"])
            filenames=[os.path.join(indir,name) for name in completed_filenames if name.split('.',1)[1]==using_suffix] 
            setnames = [os.path.basename(fname)[:-(len(using_suffix)+1)] for fname in filenames]
            #fliter_filenames=[ filename for k,filename in enumerate(filenames) if setnames[k] in filter_cells]
            fliter_filenames=[filenames[setnames.index(cell)] for k,cell in enumerate(filter_cells) if cell in setnames]
            logger.write(f'\tprocessor {rank}: computing for chromosome {chrom}', verbose_level = 1, allow_all_ranks = True)
            batch_num=100
            mat_3d=[]
            for k, filename in enumerate(fliter_filenames):
                
                csr_mat = sparse.load_npz(filename)
                #print("the th%d cell:, the mat size %d" % (k,csr_mat.shape[1]) )
                print("the chromosome: %s the th%d cell:, the mat size %d" % (chrom,k,csr_mat.shape[0]) )
                cell_mat = csr_mat.toarray()
                
                #temp= int((chrom_lens[chrom]//binsize+1)-cell_mat.shape[0])
                # if temp>0:
                #     b1=np.zeros((temp,cell_mat.shape[0]))
                #     b2=np.zeros((cell_mat.shape[0]+temp,temp))
                #     cell_mat = np.row_stack((cell_mat,b1))
                #     cell_mat = np.column_stack((cell_mat,b2)) 
                #     print("the th%d cell:, the mat size %d, %d" % (k,csr_mat.shape[0],cell_mat.shape[0]))
                    
                #     allmatrix_sp=sparse.csr_matrix(cell_mat) # 采用行优先的方式压缩矩阵
                #     sparse.save_npz(filename,allmatrix_sp)
                # else:
                #     break
                
                # temp= int((chrom_lens[chrom]//binsize+1)-cell_mat.shape[1])
                # rsize=int((chrom_lens[chrom]//binsize)+1)
                # if temp<0:
                #     cell_mat=cell_mat[:,:rsize]   
                #     print("the th%d cell:, the mat size %d, %d" % (k,csr_mat.shape[1],cell_mat.shape[1]))
                #     allmatrix_sp=sparse.csr_matrix(cell_mat) # 采用行优先的方式压缩矩阵
                #     sparse.save_npz(filename,allmatrix_sp)
                # else:
                #     break
            
                mat_3d.append(cell_mat) 
                if  (k+1)==batch_num:
                    mat_3d = np.stack(mat_3d,axis =-1)
                    cluster_results,boundaries_records,ins_strengthes_chr=Detecting_scTAD(mat_3d,window,scale)
                    print(ins_strengthes_chr.shape)
                    mat_3d=[]
                elif ((k+1)%batch_num==0):
                    mat_3d = np.stack(mat_3d,axis =-1)
                    cluster_results_temp,boundaries_records,ins_strengthes_chr_temp=Detecting_scTAD(mat_3d,window,scale)
                    cluster_results=pd.concat([cluster_results,cluster_results_temp],axis=1)
                    ins_strengthes_chr=np.vstack((ins_strengthes_chr,ins_strengthes_chr_temp))
                    print(ins_strengthes_chr.shape)
                    mat_3d=[]
            if (k+1) <batch_num:
                mat_3d = np.stack(mat_3d,axis =-1)
                cluster_results,boundaries_records,ins_strengthes_chr=Detecting_scTAD(mat_3d,window,scale)
                print(ins_strengthes_chr.shape)
                mat_3d=[]
            elif len(mat_3d)>0:
                mat_3d = np.stack(mat_3d,axis =-1)
                cluster_results_temp,boundaries_records,ins_strengthes_chr_temp=Detecting_scTAD(mat_3d,window,scale)
                cluster_results=pd.concat([cluster_results,cluster_results_temp],axis=1)
                ins_strengthes_chr=np.vstack((ins_strengthes_chr,ins_strengthes_chr_temp))
                mat_3d=[]    
            cluster_results.columns=range(len(fliter_filenames))
            np.save(os.path.join(outdir, ".".join(["ins_strengthes", chrom, "npy"])),ins_strengthes_chr)
            cluster_results.to_csv(os.path.join(outdir, ".".join(["cluster_results", chrom, "csv"])),sep = ",", index = False,header=False)
            temp_time=time.time()-t
            np.save(os.path.join(outdir, ".".join(["all_run_time", chrom, "npy"])),temp_time)
            
            
def show_boundaries_results(indir,outdir, contact_statis_pd ,chrom_lens,\
                      rank = 0, n_proc = 1,window=20):
    
    import random
    import matplotlib.pyplot as plt
    from  visual_show import show_mat_boundaries,show_mat
    proc_chroms = get_proc_chroms(chrom_lens, rank, n_proc)
    completed_filenames = os.listdir(indir)
    filter_cells=contact_statis_pd.loc[contact_statis_pd[1]>250000,0].tolist()
    for chrom in proc_chroms:
        #chrom=proc_chroms[0]
        using_suffix=".".join([chrom,  "rl", "npz"])
        filenames=[os.path.join(indir,name) for name in completed_filenames if name.split('.',1)[1]==using_suffix] 
        setnames = [os.path.basename(fname)[:-(len(using_suffix)+1)] for fname in filenames]
        
        fliter_filenames=[filenames[setnames.index(cell)] for k,cell in enumerate(filter_cells) if cell in setnames]
        
        #fliter_filenames=[ filename for k,filename in enumerate(filenames) if setnames[k] in filter_cells]
        #filter_cells=[ setname for k,setname in enumerate(setnames) if setnames[k] in filter_cells]
        cluster_results=pd.read_csv(os.path.join(outdir, ".".join(["cluster_results", chrom, "csv"])),\
                                    sep = ",", index_col = None,header=None)
            
        
           
        start=500;end=1000
        fig, axs=plt.subplots(3,3,figsize=(12,8))
        for k in range(9):
            l=k+random.randint(1,1000)
            filename=fliter_filenames[l]
            cell=filter_cells[l]
            csr_mat = sparse.load_npz(filename)
            cell_mat = csr_mat.toarray()
            submatrix=cell_mat[start:end,start:end]
            x=k//3;y=k%3
            ax=axs[x,y]
            ax.set_title(cell)
            cluster_result=cluster_results[l]
            show_mat_boundaries(submatrix,start,end,window,cluster_result,ax,fig)  
        
        
        start=500;end=1000
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        l=386
        filename=fliter_filenames[l]
        cell=filter_cells[l]
        csr_mat = sparse.load_npz(filename)
        cell_mat = csr_mat.toarray()
        submatrix=cell_mat[start:end,start:end]
        cluster_result=cluster_results[l]
        show_mat_boundaries(submatrix,start,end,window,cluster_result,ax,fig) 
        # show rl matrix
        show_mat(submatrix.T)
        
        
        
        

        
 
        
        
        