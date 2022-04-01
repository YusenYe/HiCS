#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 11:25:41 2021

@author: yusen
"""
import os
import argparse
import src.logger
from src.bin_reads_with_header import bin_sets
from src.RepresentLearning_95 import get_rl_for_all_95
from src.sc_tadboundaries import sctad_boundary
import multiprocessing
import pandas as pd


def main():
    parser = create_parser()
    args = parser.parse_args()
    chrom_dict = parse_chrom_lengths(args.chrom, args.chr_lens, args.genome, args.max_chrom_number)
    parallel_mode, rank, n_proc, parallel_properties = determine_parallelization_options(args.threaded, args.num_proc)
    
    if rank == 0 and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if parallel_mode == "parallel":
        parallel_properties['comm'].Barrier()
    threaded = True if parallel_mode == "threaded" else False
    logger = src.logger.Logger(f'{args.outdir}/snapHiC.log', rank = rank, verbose_threshold = args.verbose, threaded = threaded)
    
    
    ##step 0:statistic contact number of single cells
    import re
    def extract_setname(filename,suffix):
        return re.sub("\.*"+suffix,"",os.path.basename(filename))
    def statistic_cell_contact_num(indir):
        
        filenames=[os.path.join(indir,name) for name in os.listdir(indir)]
        setnames= [extract_setname(filename,args.suffix) for filename in filenames]
        
        contact_statis=list()
        for k,filename in enumerate(filenames):
            print(k)
            df=pd.read_csv(filename,sep='\t',header=0)
            contact_statis.append(len(df))
        
        contact_statis_pd=pd.DataFrame()
        contact_statis_pd['cell']=setnames
        contact_statis_pd['#contact']=contact_statis
        contact_statis_pd.to_csv(args.con_num,sep=',',header=None,index=None)
        return contact_statis
    if ~os.path.exists(args.indir):
        statistic_cell_contact_num(args.indir)
    
    #step 1; binning
    bin_dir = os.path.join(args.outdir, "binned")
    if 'bin' in args.steps:
        
        logger.write('starting the binning step')
        logger.flush()
        if parallel_mode == 'nonparallel':
            bin_sets(args.indir, args.suffix, binsize = args.binsize, outdir = bin_dir, \
                      chr_columns = args.chr_columns, pos_columns = args.pos_columns, \
                      low_cutoff = args.low_cutoff, n_proc = n_proc, rank = rank, logger = logger)
        elif parallel_mode == 'threaded':
            params = [(args.indir, args.suffix, args.binsize, bin_dir, args.chr_columns, args.pos_columns, \
                      args.low_cutoff, n_proc, i, logger) for i in range(n_proc)]
            with multiprocessing.Pool(n_proc) as pool:
                pool.starmap(bin_sets, params)
        logger.write("binning completed")
        logger.flush()
        #print("binned")
    
        
    #step 2 we can use network representation learning and normalization
    rl_dir = os.path.join(args.outdir, "rl_csc_95")
    if 'rl' in args.steps:
        rl_logfilename = os.path.join(rl_dir, "log.rl.txt")
        logger.write(f'Starting rl step. Additional logs for cells being processed will be written to: {rl_logfilename}')
        logger.flush()
        try:
            os.makedirs(rl_dir)
        except:
            pass
        if parallel_mode == 'nonparallel':
            rl_logfile = open(rl_logfilename, 'a')
            get_rl_for_all_95(indir = bin_dir, outdir = rl_dir, binsize = args.binsize, \
                            alpha = args.alpha, dist = args.dist, chrom_lens = chrom_dict, \
                            normalize = True, n_proc = n_proc, rank = rank, genome = args.genome, \
                            filter_file = None, parallel = False, rl_logfile = rl_logfile, \
                            rl_logfilename = rl_logfilename, threaded_lock = None, logger = logger, \
                            keep_rl_matrix = args.keep_rl_matrix)
            rl_logfile.close()
        elif parallel_mode == 'threaded':
            #rwr_logfile = open(rwr_logfilename, 'a')
            rl_logfile  = None
            from multiprocessing import Lock
            m = multiprocessing.Manager()
            threaded_lock = m.Lock()
            params = [(bin_dir, rl_dir, args.binsize, args.alpha, args.dist, chrom_dict, \
                      True, n_proc, i, args.genome, None, False, rl_logfile, \
                      rl_logfilename, threaded_lock, logger, args.keep_rl_matrix) for i in range(n_proc)]
            with multiprocessing.Pool(n_proc) as pool:
                pool.starmap(get_rl_for_all_95, params)
            #rwr_logfile.close()
        logger.write('RL computation completed for all cells')
        logger.flush()
        #print("rwr computed")
    
    
    ## step 3. detecting TAD boundaries of single cells
    window=args.window
    scale=args.scale
    tad_limit_upper=window
    contact_statis_pd=pd.read_csv(args.con_num,sep=',',header=None,index_col=None)
    sctad_dir=os.path.join(args.outdir,'sctadboundaries_'+str(scale))
    #basedir=os.path.join(args.outdir,'sctadboundaries')
    #indir = rl_dir;outdir = sctad_dir; chrom_lens = chrom_dict; binsize = args.binsize;dist = args.dist;
    #rank = rank;n_proc = n_proc; max_mem = args.max_memory; logger = logger
    if 'sctadboundaries' in args.steps:
        logger.write('starting the detection of TAD boundaries')
        logger.flush()
        if parallel_mode=="nonparallel":
            #tad_limit_upper=20
            #dist=2000000
            sctad_boundary(indir = rl_dir,outdir = sctad_dir, contact_statis_pd=contact_statis_pd,chrom_lens = chrom_dict, \
                             binsize = args.binsize, dist =args.dist,tad_limit_upper=tad_limit_upper,\
                             rank = rank, \
                             n_proc = n_proc, max_mem = args.max_memory, logger = logger,window=window,scale=scale)
        elif parallel_mode == 'threaded':
            #dist=2000000
            params = [(rl_dir, sctad_dir,contact_statis_pd, chrom_dict, \
                             args.binsize, args.dist,tad_limit_upper,\
                             i,n_proc, args.max_memory,logger,window,scale) for i in range(n_proc)]
            with multiprocessing.Pool(n_proc) as pool:
                pool.starmap(sctad_boundary,params)
        logger.write('TAD boundaries completed for all cells')
        logger.flush()
                
       
def parse_chrom_lengths(chrom, chrom_lens_filename, genome, max_chrom_number):
    if not max_chrom_number or max_chrom_number == -1:
        if not chrom or chrom == "None":
            chrom_count = 22 if genome.startswith('hg') else 19 if genome.startswith("mm") else None
            if not chrom_count:
                raise("Genome name is not recognized. Use --max-chrom-number")
            chrom = ['chr' + str(i) for i in range(1, chrom_count + 1)]
        else:
            chrom = [c.strip() for c in chrom.split()]
    else:
        chrom = ['chr' + str(i) for i in range(1, max_chrom_number + 1)]
    with open(chrom_lens_filename) as infile:
        lines = infile.readlines()
    chrom_lens = {line.split()[0]: int(line.split()[1]) for line in lines if line.split()[0] in chrom}
    return chrom_lens

def determine_parallelization_options(threaded, n_proc):
    if threaded:
        import multiprocessing
        if n_proc < 1:
            raise Exception('if threaded flag is set, n should be a positive integer')
        n_proc = n_proc
        mode = 'threaded'
        rank = 0
        properties = {}
    else:
        mode = 'nonparallel'
        n_proc = 1
        rank = 0
        properties = {}
    return mode, rank, n_proc, properties


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', action = 'store', required =False, \
                        help = 'input directory',default='Datasets_Single_mESCs')   
    parser.add_argument('-s', '--suffix', required = False, \
                        help = 'suffix of the input files', default=".abj")
    parser.add_argument('-o', '--outdir', action = 'store', \
                        required = False, help = 'output directory', default='Datasets_Single_mESCs_'+str(40)+'kb')
    parser.add_argument('-c', '--chr-columns', action = 'store', nargs = 2, \
                        type = int, help = 'two integer column numbers for chromosomes', required = False, default = [0,3])
    parser.add_argument('-p', '--pos-columns', action = 'store', nargs = 2, \
                        type = int, help = 'two integer column numbers for read positions', required = False, default = [1,4])
    parser.add_argument('-l', '--chr-lens', action = 'store', \
                        help = 'path to the chromosome lengths file', required = False,\
                           default='ext/mm9.chrom.sizes')
    parser.add_argument('-g', '--genome', action = 'store', help = 'genome name; hgxx or mmxx', \
                        required = False, default = 'mm9')
    parser.add_argument('-w', '--window', action = 'store', help = 'slide window size for computing insulation strength of boundaries', \
                        required = False, default = 20)
    parser.add_argument('--con-num', action = 'store', help = 'The directory for concact number of single cells', \
                        required = False, default = 'ext/contact_num.csv')  
    parser.add_argument('--scale', action = 'store', help = 'scale parameter for hierarcgucal chromatin strucutres', \
                    required = False, default = 1) 
    parser.add_argument('--chrom', action = 'store', help = 'chromosome to process', \
                        required = False, default = None)
    parser.add_argument('--binsize', type = int, help = 'bin size used for binning the reads', \
                        required = False, default = 4e4)
    parser.add_argument('--low-cutoff', type = int, help = 'cut-off for removing short-range reads', \
                        default = 4e4, required = False)     
    parser.add_argument('--alpha', type = float, help = 'reserve the top edges for downstream analysis', \
                        default = 0.05, required = False)
    parser.add_argument('--dist', type = int, help = 'distance from diagonal to consider', \
                        default =2000000, required = False)
    parser.add_argument('--threaded', action = 'store_true', default =False, \
                        help = 'if set, will attempt to use multiprocessing on single machine', required = False)
    parser.add_argument('-n', '--num-proc', help = 'number of processes used in threaded mode',
                        required = False, default = 0, type = int)
    parser.add_argument('--max-memory', default = 12, type = float, required = False, \
                        help = 'memory available in GB, that will be used in constructing dense matrices')
    parser.add_argument('--verbose', type = int, required = False, default = 0,
                        help = 'integer between 0 and 3 (inclusive), 0 for the least amount to print to log file') 
    parser.add_argument('--steps', nargs = "*", default = ['bin','rl','sctadboundaries'], \
                        required = False, help = 'steps to run. Default is all steps.')
    parser.add_argument('--keep-rl-matrix', action = 'store_true', default = False, \
                        help = 'if set, will store the computed rwr matrix in entirety in npy format', required = False)
    parser.add_argument('--max-chrom-number', action = "store", required = False, type = int, \
                        help = "biggest chromosome number to consider in genome, for example 22 for hg", default = -1)
    return parser

if __name__ == "__main__":
    main()
    #pass
    






