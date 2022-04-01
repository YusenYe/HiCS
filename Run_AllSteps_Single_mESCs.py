# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 17:38:27 2022

@author: user
"""


import os

#package dir
package_dir='E:/teacher-2020/Projects/Project11/Codes/HiCS_Packages/'
os.chdir(package_dir)
######################################################################
#################    Users Variables    ##############################
######################################################################
#Dataset location
#output locations
BINSIZE=40000
INDIR='Datasets_Single_mESCs'
OUTDIR='Datasets_Single_mESCs_'+str(BINSIZE//1000)+'kb'   
#the name of reference genome
#GENOME='mm9';
CHR_LEN='ext/mm9.chrom.sizes';
###file suffix
#FILE_SUFFIX=".abj"
###statistic concact counts of single cells
#CONCACT_NUM=package_dir+'ext/contact_num.csv'
#LOW_CUTOFF=1e4
#CHR_COLUMNS=[0,3]
#POS_COLUMNS=[1,4]
###slide window size of computing insulation strength
#WINDOW=20
########################################################################
SCALE=1
command='python HiCS_main.py -i '+INDIR+' -o '+OUTDIR+' -l '+CHR_LEN+' --binsize '+str(BINSIZE)+' --scale '+str(SCALE)
os.system(command)



# ########################################################################
# ###########            run step 1 binning                  #############
# ########################################################################
# #command='python HiCS_main.py -i INDIR -o OUTDIR -l --binsize BINSIZE --step steps'
# STEPS="bin"
# command='python HiCS_main.py -i '+INDIR+' -o '+OUTDIR+' -l '+CHR_LEN+' --binsize '+str(BINSIZE)+' --steps '+STEPS
# os.system(command)


# ########################################################################
# ###########           step 2:imputing                      #############
# ########################################################################
# #command='python HiCS_main.py -i INDIR -o OUTDIR -l --binsize BINSIZE --step steps'
# STEPS="rl"
# command='python HiCS_main.py -i '+INDIR+' -o '+OUTDIR+' -l '+CHR_LEN+' --binsize '+str(BINSIZE)+' --steps '+STEPS
# os.system(command)


# ########################################################################
# #####  run step 3:detecting TAD boundaries of single cells       #######
# ########################################################################
# #command='python HiCS_main.py -i INDIR -o OUTDIR -l --binsize BINSIZE --step STEPS --scale SCALE'
# #please setting scale parameter
# SCALE=1
# STEPS='sctadboundaries'
# command='python HiCS_main.py -i '+INDIR+' -o '+OUTDIR+' -l '+CHR_LEN+' --binsize '+str(BINSIZE)+' --scale '+str(SCALE)+' --steps '+STEPS
# os.system(command)






