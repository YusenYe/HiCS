# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 16:33:52 2021

@author: user
"""

import pandas as pd
import numpy as np
import scipy as sp
import networkx as nx
import os
import re
import sys
import random
from collections import deque
import gc
#from src.node2vec import Node2vec
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

#@profile
#edge_filename=filepath
def get_rl_en(edge_filename, binsize , distance , chrom , chrom_len , \
                                                alpha , final_try = False, logfile = None, parallel = False, threaded_lock = None):
    gc.collect()
    setname = edge_filename[(edge_filename.rfind('/') + 1):]
    #print('computing rl for', setname, chrom)
    #sys.stdout.flush()
    edgelist = pd.read_csv(edge_filename, sep = ",", header = None, names = ["chr1", "x1", "x2", "chr2", "y1", "y2", "weight"])
    edgelist = edgelist[(edgelist['chr1'] == chrom) & (edgelist['chr1'] == edgelist['chr2'])]
    edgelist = bin_matrix(edgelist, binsize)
    NUM = int(np.ceil(chrom_len / binsize))
    edgelist=edgelist.loc[ (edgelist['x1']<NUM) & (edgelist['y1']<NUM)]
    #if np.max((edgelist['x1'].max(),edgelist['y1'].max())) <NUM:
    #print('NUM', NUM)
    #edges = pd.DataFrame({'x1':list(range(0, NUM-1)), 'y1':list(range(1, NUM))})
    edges = pd.DataFrame({'x1':list(range(0, NUM)), 'y1':list(range(0, NUM))})
    edges = pd.concat([edges, edgelist[['x1', 'y1']]], axis = 0)
    #edges.drop_duplicates(inplace=True)
    edges.loc[:,'weight'] = 1
    g = nx.from_pandas_edgelist(edges, source = 'x1', target = 'y1', edge_attr = ['weight'], create_using = nx.Graph())
    model = Node2Vec(g, dimensions=32,walk_length=5,num_walks=50,workers=4,p=1,q=1)  # Use temp_folder for big graphs
    model = model.fit(window=5)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    #model.wv.most_similar('200.0')
    embeddings = {}
    for word in g.nodes():
        embeddings[word] = model.wv[str(word)] 
    
    similarity=cosine_similarity(pd.DataFrame(embeddings).T)
    #print('deleting edgelist', setname, chrom)
    #sys.stdout.flush()
    del edgelist
    gc.collect()
    del g, edges
    gc.collect()
    return similarity,embeddings
    #else:
    #    return np.array([0]),np.array([0])


def bin_matrix(df, binsize):
    df.loc[:,'x1'] = df.loc[:,'x1'] // binsize
    df.loc[:,'y1'] = df.loc[:,'y1'] // binsize
    return df

def get_stochastic_matrix_from_edgelist(edgelist):
    g = nx.from_pandas_edgelist(edgelist, source = 'x1', target = 'y1', edge_attr = ['weight'], create_using = nx.Graph())
    degrees = np.array([g.degree(i) for i in g.nodes()])
    m = sp.sparse.csc_matrix(nx.adjacency_matrix(g).astype(float))
    m.data /= degrees[m.indices] #stochastic matrix
    del g, degrees
    return m

def reformat_sparse_matrix(m, binsize, distance):
    max_bin_distance = distance // binsize
    df = pd.DataFrame({'x1':m.row, 'y1':m.col, 'value':m.data})
    df = df[((df['y1'] - df['x1']) <= max_bin_distance) & ((df['y1'] - df['x1']) > 0)]
    df.iloc[:,0] = df.iloc[:,0] * binsize
    df.iloc[:,1] = df.iloc[:,1] * binsize
    df.loc[:,'x2'] = df['x1'] + binsize
    df.loc[:,'y2'] = df['y1'] + binsize
    df[['x1', 'x2', 'y1', 'y2']] = df[['x1', 'x2', 'y1', 'y2']].astype(int) 
    return df

def normalize_along_diagonal(d, trim = 0.01):
    #trim_value = d['value'].quantile(1-trim)
    vals = d['value'].tolist()
    vals.sort()
    vals.reverse()
    trim_value = vals[round(trim * len(vals)) - 1]
    #print(trim_value)
    remaining = d[d['value'] <= trim_value]['value']
    mu = np.mean(remaining)
    sd = np.std(remaining)
    #print(trim_value, remaining.shape[0], mu, sd)
    d.loc[:,'value'] = (d['value'] - mu) / sd
    return d

def determine_proc_share(indir, chrom_lens, n_proc, rank, outdir, ignore_sets = set(), rewrite = True):
    filenames = os.listdir(indir)
    filenames = [name for name in filenames if name.endswith(".bedpe")]
    filenames.sort()
    #print(filenames)
    setnames = [os.path.basename(fname)[:-len(".bedpe")] for fname in filenames]
    chrom_list = [(k, chrom_lens[k]) for k in list(chrom_lens.keys())]
    chrom_list.sort(key = lambda x: x[1])
    chrom_list.reverse()
    chrom_names = [i[0] for i in chrom_list]
    #chrom_names = list(chrom_lens.keys())
    #if rank == 0:
    #    print('sorted chroms', chrom_names)
    #chrom_names.sort()
    jobs = [(chrom_names[i], filenames[j], setnames[j]) for i in range(len(chrom_names)) for j in range(len(filenames))]
    #print('init jobs len', len(jobs), '.ignore sets len', len(ignore_sets))
    jobs = [job for job in jobs if (job[2], job[0]) not in ignore_sets]
    #print('jobs len after removing ignores', len(jobs), jobs[0])
    #print('example ignore', list(ignore_sets)[0] if len(ignore_sets) > 0 else "0")
    #print('init jobs len', len(jobs))
    if rewrite:
        completed_filenames = os.listdir(outdir)
        #print(completed_filenames[0])
        incompl = [name for name in completed_filenames if not name.endswith(".rl.npz")]
        if rank == 0:
            pass
            #print(incompl)
        completed_setnames = [name[:-len(".rl.npz")] for name in completed_filenames if name.endswith(".rl.npz")]
        #print(completed_filenames[0])
        #print('filenames', len(completed_filenames))
        completed_filenames.sort()
        #completed_setnames = [re.search(r".*\..*?\.", fname).group()[:-1] for fname in completed_filenames]
        #print('completed_senames', len(completed_setnames))
        #print(completed_setnames[0])
        completed_pairs = set([(setname[:setname.rfind('.')], setname[(setname.rfind('.') + 1):]) for setname in completed_setnames])
        #print(len(completed_pairs), 'example completed', list(completed_pairs)[0] if len(completed_pairs) > 0 else "0")
        if False and rank == 0:
            print('jobs vs complted', len(jobs), len(completed_pairs))
            #print(completed_pairs[:2])
            print(jobs[0])
            print(list(completed_pairs)[0])
        jobs = [job for job in jobs if (job[2], job[0]) not in completed_pairs]
        if False and rank == 0:
            print('new jobs len', len(jobs))
    jobs.sort()
    indices = list(range(rank, len(jobs), n_proc))
    #random.seed(4)
    #random.shuffle(jobs)
    #random.seed()
    #if rank in [1,2,3, 19]:
    #    print(rank, [(i[0], i[2]) for i in jobs[:5]])
    proc_jobs = [jobs[i] for i in indices]
    #proc_jobs = deque(proc_jobs)
    #proc_jobs.rotate(rank)
    #proc_jobs = list(proc_jobs)
    random.shuffle(proc_jobs)
    #print('jobs for rank', rank, len(proc_jobs), 'from total', len(jobs));print(n_proc, rank);
    return proc_jobs
    
def ammend_ignore_list(logfile, ignore_filename):
    ignore = []
    if logfile:
        with open(logfile, 'r') as lfile:
            lines = lfile.readlines()
        lines = [line.split() for line in lines]
        started = set()
        completed = set()
        for line in lines:
            if len(line) < 5:
                continue
            if line[0] == "solved":
                #remove .bedpe suffix from setname
                completed.add((line[-2], line[-1][:-6]))
            else:
                started.add((line[-2], line[-1]))
        ignore = started.difference(completed)
        with open(ignore_filename, 'a') as ofile:
            for chrom, setname in ignore:
            	ofile.write(f'{setname} {chrom}\n')
        
def get_ignore_list(filename):
    with open(filename) as ifile:
        lines = ifile.readlines()
    lines = [line.split() for line in lines]
    ignores = set()
    for line in lines:
        ignores.add((line[0], line[1]))
    return ignores

def get_nth_diag_indices(mat, offset):
    rows, cols_orig = np.diag_indices_from(mat)
    cols = cols_orig.copy()
    if offset > 0:
        cols += offset
        rows = rows[:-offset]
        cols = cols[:-offset]
    return rows, cols

def normalize_along_diagonal_from_numpy(d, chrom, max_bin_distance, output_filename, binsize):
    #df_all = pd.DataFrame()
    if os.path.exists(output_filename):
        os.remove(output_filename)
    with open(output_filename, "a") as f:
        for offset in range(1, max_bin_distance + 1):
            r, c = get_nth_diag_indices(d, offset)
            vals_orig = d[r,c].tolist()
            if isinstance(vals_orig[0], list):
                vals_orig = vals_orig[0]
            df = pd.DataFrame({'x1': r, 'y1': c, 'v': vals_orig})
            df['x1'] = ((df['x1'] ) * binsize).astype(int)
            df['y1'] = ((df['y1'] ) * binsize).astype(int)
            df['x2'] = (df['x1'] + binsize).astype(int)
            df['y2'] = (df['y1'] + binsize).astype(int)
            df['chr1'] = chrom
            df['chr2'] = chrom
            df = df[['chr1', 'x1', 'x2', 'chr2', 'y1', 'y2', 'v']]
            df.to_csv(f, mode='a', header=False, sep = ",", index = False)

def keep_eligible_distance(d, dist, binsize):
    irange = d.shape[0]
    jrange = int(dist // binsize)
    inds = [i for i in range(irange) for j in range(i, i+jrange)]
    indps = [j for i in range(irange) for j in range(i+1, i+1+jrange)]
    keep_matrix = sp.sparse.csr_matrix(([1 for i in range(len(inds))], (inds, indps)))
    keep_matrix = keep_matrix[:,:d.shape[1]]
    #print(keep_matrix[0, 198:208])
    d = keep_matrix.multiply(sp.sparse.csr_matrix(d))
    #print(d[0, 198:208])
    return d

def get_rl_for_all_95(indir, outdir , binsize, alpha , dist, chrom_lens , 
                normalize = False, n_proc = 1, rank = 0, genome = 'mouse', filter_file = None, parallel = False, 
                                rl_logfile = None, rl_logfilename = None, threaded_lock = None, logger = None,
                                 keep_rl_matrix = False):
    if logger:
        logger.set_rank(rank)
    
    if not outdir:
        outdir = indir
        outdir = os.path.join(outdir, "rl")
    try:
        os.makedirs(outdir)
    except:
        pass
    if rl_logfile is None:
        rl_logfile = open(rl_logfilename, 'a') if rl_logfilename else None
    processor_jobs = determine_proc_share(indir, chrom_lens, n_proc, rank, outdir)#, ignore_sets = ignore_sets)
    retry_filename = os.path.join(outdir, ('_'.join([str(rank), "retry", "instances"]) + ".txt"))
    attempt_counter = 0
    attempts_allowed = 10
    #print(rank, [(name[0], name[2]) for name in processor_jobs])
    #sys.stdout.flush()
    logger.write(f'\tprocessor {rank}: {len(processor_jobs)} jobs assigned to processor {rank}.', \
                             append_time = False, allow_all_ranks = True, verbose_level = 2)
    while len(processor_jobs) > 0 and attempt_counter < attempts_allowed:
        #print(rank, 'has len', len(processor_jobs), attempt_counter)
        #sys.stdout.flush()
        for chrom, filename, setname in processor_jobs:
            print(filename)
            #logger.flush()
            gc.collect() #garbage collection
            
            filepath = os.path.join(indir, filename)
            
            d,emb = get_rl_en(filepath, binsize = binsize, distance = dist, chrom = chrom, chrom_len = chrom_lens[chrom], 
                alpha = alpha, logfile = rl_logfile, parallel = parallel, threaded_lock = threaded_lock)
            
            if d.shape[0]>1:
                #print(chrom,d.shape)
                thr=np.percentile(d,100-alpha*100)
                d[d<thr]=0
                d= np.triu(d, 1)
                
                allmatrix_sp=sparse.csr_matrix(d) # 采用行优先的方式压缩矩阵
                
                output_filename = os.path.join(outdir, ".".join([setname, chrom, "rl",'npz']))
                
                sparse.save_npz(output_filename,allmatrix_sp)
                csr_mat = sparse.load_npz(output_filename)
                
                output_filename_emb=os.path.join(outdir, ".".join([setname, chrom, "embvector",'npy']))
                #sparse.save(output_filename_emb,emb)
                #emb_temp=sparse.load_npz(output_filename_emb)
                np.save(output_filename_emb, emb)
                # new_dict = np.load(output_filename_emb, allow_pickle='TRUE')
                # print(new_dict)
                # keys = emb.keys()  
                # values = emb.values()  
                # df = pd.DataFrame({'KEYS':keys,'VALUES':values})  
                # df.to_csv(output_filename_emb,index=False) 
                #df=pd.read_csv(output_filename_emb,index_col=None,header=0)
                print("%s, the mat size: %d, %d" % (chrom,d.shape[0],csr_mat.shape[0]))
            else:
                print("error")
                
        if os.path.exists(retry_filename):
            logger.write(f'\tprocessor {rank}: Attempting to re-run failed jobs. Attempt: {attempt_counter + 1}/{attempts_allowed}', \
                              append_time = False, allow_all_ranks = True, verbose_level = 3)
            #print("rank", rank, ": attempting to rerun failed jobs. Attempt #", attempt_counter + 1)
            #sys.stdout.flush()
            with open(retry_filename, 'r') as infile:
                jobs = infile.readlines()
            jobs = [line.split() for line in jobs]
            processor_jobs = jobs
            os.remove(retry_filename)
            attempt_counter += 1
        else:
            processor_jobs = []
            logger.write(f'\tprocessor {rank}: finished processing my share (RL)', \
                              append_time = False, allow_all_ranks = True, verbose_level = 3)
            #print("rank", rank, ": no remaining jobs or parser failed")
            #sys.stdout.flush()
    if attempt_counter == attempts_allowed:
        logger.write(f'\tprocessor {rank}: Failed to finish assigned jobs after {attempts_allowed} attempts.')
        #return df
    if rl_logfile and not parallel:
        rl_logfile.close()

if __name__ == "__main__":
    pass
    



        
        
        
        
        
        
        
        
    
    
    


