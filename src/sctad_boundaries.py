#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:15:37 2021

@author: yusen
"""
import os
import numpy as np
import scipy as sp
# from scipy import stats
import pandas as pd
# import skimage
# import subprocess
# from statsmodels.stats.multitest import multipletests
# import gc
# import sys
import h5py
eps=1e-20
MAX=1e20
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


def determine_dense_matrix_size(num_cells, dist, binsize, max_mem):
    max_mem_floats = max_mem * 1e9 
    max_mem_floats /= 8
    square_cells = max_mem_floats // num_cells
    mat_size = int(np.floor(np.sqrt(square_cells)) / 4)
    #print(mat_size)
    mat_size = max(int((dist // binsize) + 50), mat_size)
    #print('mat_size:', mat_size)
    #if mat_size < (dist // binsize):
    #    raise "Specified " + str(max_mem) + "GB is not enough for constructing dense matrix with distance " + str(dist) + "."
    return mat_size


def convert_sparse_dataframe_to_dense_matrix(d, mat_size, dist, binsize, window, num_cells, chrom_size, chrom_filename):
    d['i'] = (d.iloc[:,1] // binsize).astype(int)
    d['j'] = (d.iloc[:,4] // binsize).astype(int)
    #all_rows = set(range(d.shape[0]))
    #max_distance_bin = dist // binsize
    chrom_bins = int(chrom_size // binsize)
    for i in range(0, chrom_bins + 1, int(mat_size)):
        matrix_upper_bound = max(0, i - window)
        matrix_lower_bound = min(i + mat_size + window, chrom_bins + 1)
        #print (i, matrix_upper_bound,matrix_lower_bound)
        keeprows = list(np.where((d['i'] >= matrix_upper_bound) & (d['j'] < matrix_lower_bound))[0])
        d_portion = d.iloc[keeprows, 0:6].reset_index(drop = True)
        d_portion.columns = ['chr1','x1','x2','chr2','y1','y2']
        #print(d_portion, 'd_portions shape')
        #skiprows = all_rows.difference(keeprows)
        hdf_file = h5py.File(chrom_filename + '.cells.hdf', 'r')
        portion = hdf_file[list(hdf_file.keys())[0]]
        #print(type(keeprows))
        #if isinstance(keeprows, list):
        #    print(len(keeprows))
        #    if len(keeprows) > 0:
        #        print(keeprows[0], type(keeprows[0]))
        if len(keeprows) == 0:
            continue
        portion = portion[keeprows, :]
        hdf_file.close()
        if portion.shape[0] == 0:
            continue
        portion = pd.DataFrame(portion)
        #print('portions shape', portion.shape)
        portion = pd.concat([d_portion, portion], axis = 1)
        #print('concatted portion', portion.shape)
        #print(np.where(portion.isnull().sum() > 0))
        portion['i'] =  (portion.loc[:,'x1'] // binsize).astype(int)
        portion['j'] =  (portion.loc[:,'y1'] // binsize).astype(int)
        #portion_old = d[(d['i'] >= matrix_upper_bound) & \
        #            (d['j'] < matrix_lower_bound)]
        #if portion.shape[0] == 0:
        #    continue
        full_sparse = pd.DataFrame({'i': range(min(portion['i']), max(portion['j'])-1), \
                    'j': range(min(portion['i'])+1, max(portion['j']))})
        portion = portion.merge(full_sparse, on = ['i','j'], how = "outer")
        #print(portion.iloc[:,list(range(7)) + [11, 12]])
        #print("start", matrix_upper_bound, "end", matrix_lower_bound)
        dense_cells = []
        #print('here', portion.columns)
        #print(portion.head())
        #print(portion.dtypes[:20])
        #sys.stdout.flush()
        for cell_index in range(num_cells):
            cell_mat = sp.sparse.csr_matrix((portion.iloc[:, 6 + cell_index], \
                                            ((portion['i'] - matrix_upper_bound), \
                                             (portion['j'] - matrix_upper_bound))), \
                                           shape = (matrix_lower_bound - matrix_upper_bound, \
                                                    matrix_lower_bound - matrix_upper_bound))
            cell_mat = np.array(cell_mat.todense())
            cell_mat[np.tril_indices(cell_mat.shape[0],0)] = np.nan
            dense_cells.append(cell_mat)
        mat_3d = np.stack(dense_cells, axis = -1)
        if matrix_upper_bound == 0:
            pad_size = abs(i - window)
            mat_3d = np.pad(mat_3d, ((pad_size, 0), (pad_size, 0), (0, 0)), mode = 'constant', constant_values = np.nan)
        if matrix_lower_bound == chrom_bins + 1:
            pad_size = window #- 1
            mat_3d = np.pad(mat_3d, ((0, pad_size), (0, pad_size), (0, 0)), mode = 'constant', constant_values = np.nan)
        yield mat_3d, i

#mat=submatrix; limit=tad_limit_upper
def compute_boundaries(mat,window=20):
    ins_strengthes=pd.DataFrame()
    mat_size=mat.shape
    mat[(np.isnan(mat))]=0
    mat[mat<0]=0
    for i in range(window,mat_size[0]-window):
        #print(i)
        #compute insulation score
        #mat[(np.isnan(mat))]=0
        gap=mat[i,i:,:].sum(axis=0)+mat[:i,i,:].sum(axis=0)
        gap[gap>0]=1
        ins_b=mat[i-window:i,i+1:i+window+1,:].sum(axis=0).sum(axis=0)
        #ins_b=ins_b*gap
        ins_a_1=mat[i-window:i,i-window:i,:].sum(axis=0).sum(axis=0)
        ins_a_2=mat[i+1:i+window+1,i+1:i+window+1,:].sum(axis=0).sum(axis=0)
        ins_b=ins_b*gap
        # if ins_b.any()==0:
        #     print('ins_b:',i)
        if (ins_a_1.any()==0) or (ins_a_2.any()==0):
            ins_a=ins_a_1*ins_a_2
            ins_a[ins_a>0]=1
            ins_b=ins_b*ins_a
        ins_strength=(ins_a_1+ins_a_2-ins_b+eps)/(ins_b+ins_a_1+ins_a_2+eps)
        #ins_strength[ins_strength==1]=np.nan
        #print(ins_a_1[9],ins_a_2[9],ins_b[9])
        ins_strengthes[i]=ins_strength
        #print(i,ins_strength[9])    
    return ins_strengthes



#vector=ins_vector;limit=tad_limit_upper 
#set max search range max_dis=500              
def detect_can_boundaries(vector,max_dis=500):
    hdmd_vector=np.empty(vector.shape[0])
    vector=vector+(np.random.rand(vector.shape[0])/MAX)
    for loc in range(vector.shape[0]):
        element=vector[loc]
        if element!=1:
            upper_limit=max(-1,loc-max_dis-1)
            flag1=0
            for left in range(loc-1,upper_limit,-1):
                if vector[left]>=element:
                    flag1=1
                    break
            flag2=0
            lower_limit=min(vector.shape[0],loc+max_dis+1)
            
            for right in range(loc+1,lower_limit):
                if vector[right]>=element:
                    flag2=1
                    break
            #higher density minmun distance point
            if (flag1==0) and (flag2!=0):
                hdmdp=right-loc
            elif (flag1!=0) and (flag2==0):
                hdmdp=loc-left
            elif (flag1!=0) and (flag2!=0):
                hdmdp=min(loc-left,right-loc)
            else:
                hdmdp=max_dis
            hdmd_vector[loc]= hdmdp
        else:
            hdmd_vector[loc]= np.nan 
    return hdmd_vector
    #showing the result

def cluster_result(boundaries_record,scale=1):
    #temp_den=boundaries_record['den']/max(boundaries_record['den'])
    #temp_hdmd=boundaries_record['hdmd']/max(boundaries_record['hdmd'])
    #boundaries_record['den*hdmd']=temp_den*temp_hdmd
    boundaries_record['den*hdmd']=boundaries_record['den']*boundaries_record['hdmd']
    boundaries_record['rank']=boundaries_record['den*hdmd'].rank(ascending = False, method = 'dense')
    
    temp_rank = boundaries_record['rank'] / np.nanmax(boundaries_record['rank'])
    #print(candidates['eta'].describe())
    temp_eta = boundaries_record['den*hdmd'] / np.nanmax(boundaries_record['den*hdmd'])
    
    boundaries_record['transformed_den*hdmd'] =  (temp_eta + temp_rank)/np.sqrt(2)
    #plt.plot(candidates['transformed_rank'], candidates['transformed_eta'], '.')
    #print(candidates.shape, ':canidates shape')
    #print(candidates['transformed_eta'].idxmin(), 'idxmin of transformed eta')
    breakpoint = boundaries_record.iloc[boundaries_record['transformed_den*hdmd'].idxmin()]['den*hdmd']
    breakpoint=breakpoint*scale
    boundaries_record['cluster']=-1
    boundaries_record.loc[ (boundaries_record['den*hdmd']>breakpoint) ,\
                          'cluster'] = boundaries_record.loc[boundaries_record['den*hdmd']>breakpoint, \
                        'cluster'].rank(ascending = False, method = 'first')
    #set all cluster centers set 1,other set 0
    #plt.plot(boundaries_record['rank'],boundaries_record['den*hdmd'],'.')
    #plt.vlines(1065,0,1)
    return boundaries_record
    


def sctad_boundary(indir, outdir, chrom_lens, binsize, dist, tad_limit_upper, \
                      rank = 0, n_proc = 1, max_mem = 2, logger = None, window=20):
    """
    detecting TAD boundaries from single cells

    """
    logger.set_rank(rank)
    try:
        os.makedirs(outdir)
    except:
        pass
    proc_chroms = get_proc_chroms(chrom_lens, rank, n_proc)
    #print(rank, proc_chroms)
    #sys.stdout.flush()
    #chrom=proc_chroms[0]
    for chrom in proc_chroms:
        #chrom='chr19'
        logger.write(f'\tprocessor {rank}: computing for chromosome {chrom}', verbose_level = 1, allow_all_ranks = True)
        #print(rank, chrom)
        #d = pd.read_csv(chrom_filename, sep = "\t", header = None, usecols = [0,1,2,3,4,5, num_cells + 6])
        ##command = "awk -F '\t' '{print NF; exit}' " + chrom_filename
        ##proc_output = subprocess.check_output(command, shell = True, executable = "/bin/bash")
        ##num_cells = int(proc_output) - 7
        if ~os.path.exists(os.path.join(outdir, ".".join(["ins_strengthes", chrom, "npy"]))):
            chrom_filename = os.path.join(indir, ".".join([chrom, "normalized", "combined", "bedpe"]))
            with h5py.File(chrom_filename + ".cells.hdf", 'r') as ifile:
                num_cells = ifile[chrom].shape[1]
            logger.write(f'\tprocessor {rank}: detected {num_cells} cells for chromosome {chrom}', \
                                 append_time = False, allow_all_ranks = True, verbose_level = 2)
            #print('num_cells', num_cells)
            #sys.stdout.flush()
            d = pd.read_csv(chrom_filename, sep = ",", header = None)
            #num_cells = d.shape[1] - 7
            matrix_max_size = determine_dense_matrix_size(num_cells, dist, binsize, max_mem)
            #print(rank, matrix_max_size)
            submatrices = convert_sparse_dataframe_to_dense_matrix(d, matrix_max_size, \
                                                                   dist, binsize, window, \
                                                                   num_cells, chrom_lens[chrom], chrom_filename) 
            #ins_strengthes_chr=pd.DataFrame()
            #chrom_len=int(chrom_lens[chrom]//binsize) +1
            for k, (submatrix, start_index) in enumerate(submatrices):
                #print(start_index,submatrix.shape)
                logger.write(f'\tprocessor {rank}: computing background for batch {k} of {chrom}, start index = {start_index}', \
                                  verbose_level = 3, allow_all_ranks = True, append_time = False)
                
                ins_strengthes=compute_boundaries(submatrix,window)
                #ins_strengthes.loc[:,(ins_strengthes.columns>=tad_limit_upper) or (ins_strengthes.columns>=tad_limit_upper )]
                if k==0:
                    ins_strengthes_chr=ins_strengthes.to_numpy()
                else:
                    ins_strengthes_chr=np.concatenate((ins_strengthes_chr,ins_strengthes.to_numpy()),axis=1)
            #ins_strengthes_chr=ins_strengthes_chr[:,tad_limit_upper:(chrom_len-tad_limit_upper)]
            np.save(os.path.join(outdir, ".".join(["ins_strengthes", chrom, "npy"])),ins_strengthes_chr)
        else:
            ins_strengthes_chr=np.load(os.path.join(outdir, ".".join(["ins_strengthes", chrom, "npy"])))
        
        #detecting　candidated TAD boundaries of each single cells
        cluster_results=pd.DataFrame()
        for i in range(num_cells):
            #print(i)
            ins_vector=ins_strengthes_chr[i,:]
            vector=ins_vector[tad_limit_upper:(ins_vector.shape[0]-tad_limit_upper)]
            hdmd_vector=detect_can_boundaries(vector,max_dis=500)
        
            boundaries_record=pd.DataFrame()
            boundaries_record['den']=vector
            boundaries_record['hdmd']=hdmd_vector
            boundaries_record=cluster_result(boundaries_record)
            
            gap=vector.copy()
            gap[gap!=1]=0
            gap[gap==1]=np.nan
            cluster_results[i]=boundaries_record['cluster']+gap
            print(chrom,i,':',cluster_results.shape[0])
            #boundaries_record['cluster']=boundaries_record['cluster']+gap
            #filtering boundaries using file
        #boundaries_record.to_csv(os.path.join(outdir, ".".join(["boundaries_record", chrom, "csv"])),sep = ",", index = False,header=False)
        cluster_results.to_csv(os.path.join(outdir, ".".join(["cluster_results", chrom, "csv"])),sep = ",", index = False,header=False)
        
            
        
            
            


    
    
    
    
    
    
    
    
            
        
        
        

                    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                