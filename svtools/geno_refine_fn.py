#!/usr/bin/env python

import argparse, sys, copy, gzip, time, math, re
import numpy as np
import pandas as pd
import fastcluster
from scipy import stats, cluster, spatial
from sklearn import metrics
from collections import Counter, defaultdict, namedtuple
import statsmodels.formula.api as smf
from operator import itemgetter
import warnings
from svtools.vcf.file import Vcf
from svtools.vcf.genotype import Genotype
from svtools.vcf.variant import Variant
import svtools.utils as su

vcf_rec = namedtuple ('vcf_rec', 'var_id sample svtype AF GT CN AB')

def adjust_mismatches(df, dist_sq, a, b):

    df_a=df.loc[(df['gtn']==a) & (df['gt_new']==a)]
    df_b=df.loc[(df['gtn']==b) & (df['gt_new']==b)]
    df_mismatch=df.loc[((df['gtn']==a) & (df['gt_new']==b)) | ((df['gtn']==b) & (df['gt_new']==a))]
    if df_mismatch.shape[0]==0 or (df_a.shape[0]==0 and df_b.shape[0]==0):
        return
    sample_map={key: index for index,key in enumerate(df['sample'])}
    for samp in df_mismatch.loc[:, 'sample']:
        if df_a.shape[0]==0:
            gp=b
        elif df_b.shape[0]==0:
            gp=a
        else:    
            da=min(dist_sq[df_a.loc[:,'sample'].map(sample_map), sample_map[samp]])
            db=min(dist_sq[df_b.loc[:,'sample'].map(sample_map), sample_map[samp]])
            gp=(a if da<db else b)
        df.loc[sample_map[samp], 'gt_adj']=gp
    return 


def recluster(df):
    df.to_csv('./df_a.csv')
    df=df[(df.AB!=".")].copy()
    
    df.loc[:,'AB']=pd.to_numeric(df.loc[:,'AB'])
    df.loc[:,'CN']=pd.to_numeric(df.loc[:,'CN'])

    tp=df.iloc[0,:].loc['svtype']

    [mn_CN, mn_AB]=df.loc[:, ['CN', 'AB']].mean()
    [sd_CN, sd_AB]=df.loc[:, ['CN', 'AB']].std()
    df.loc[:, 'CN1']=(df.loc[:,'CN']-mn_CN)/sd_CN
    df.loc[:, 'AB1']=(df.loc[:,'AB']-mn_AB)/sd_AB
    
    gt_code={'0/0':1, '0/1':2, '1/1':3}
    df.loc[:,'gtn']=df.loc[:, 'GT'].map(gt_code)
    cc=['gtn', 'clus2D_3', 'clus1D_3', 'clus2D_2', 'clus1D_2']

    dist_2d=spatial.distance.pdist(df[['AB1', 'CN1']], metric='cityblock')
    dist_1d=spatial.distance.pdist(df[['AB1']], metric='cityblock')

    clus_2d=fastcluster.linkage(dist_2d, method='average', preserve_input='True')
    clus_1d=fastcluster.linkage(dist_1d, method='average', preserve_input='True')
    
    df.loc[:,'clus2D_3']=cluster.hierarchy.fcluster(clus_2d, 3, criterion='maxclust')
    df.loc[:,'clus2D_2']=cluster.hierarchy.fcluster(clus_2d, 2, criterion='maxclust')
    df.loc[:,'clus1D_3']=cluster.hierarchy.fcluster(clus_1d, 3, criterion='maxclust')
    df.loc[:,'clus1D_2']=cluster.hierarchy.fcluster(clus_1d, 2, criterion='maxclust')

    dist_2d_sq=spatial.distance.squareform(dist_2d)
    dist_1d_sq=spatial.distance.squareform(dist_1d)

    sil={}
    for clus in cc:
      sil[clus]=metrics.silhouette_score(dist_2d_sq, df.loc[:, clus].values, metric='precomputed')

    [min_AB, max_AB]=[df.loc[:, 'AB'].min(), df.loc[:, 'AB'].max()]

    if min_AB>0.01:
      sys.stderr.write("skipping\n")
    else:
        nclus=0
        if max_AB<0.6:
            nclus=2
        elif max_AB>0.75:
            nclus=3
        else:
            nclus=2
            sys.stderr.write("2 o4 3\n")
        refine_clusters_2D(df, dist_1d_sq, sil, nclus)
    return df

def refine_clusters_2D(df, dist_sq, sil, nclus):

    [clus_1d, clus_2d]=['clus1D_3', 'clus2D_3']
    if nclus==2:
        [clus_1d, clus_2d]=['clus1D_2', 'clus2D_2']
    elif nclus!=3:
        sys.stderr.write("Error:  Number of clusters is not 2 or 3.\n")
        sys.exit(1)
    if (sil[clus_2d]<0.65 and sil[clus_1d]<0.65):
        sys.stderr.write("bad clusters\n")
    elif sil[clus_2d]>0.7:
        refine_2D(df, clus_2d, dist_sq, nclus)
    elif sil[clus_1d]>0.7:
        refine_2D(df, clus_1d, dist_sq, nclus)
    else:
        clus=(clus_2d if sil[clus_2d]>sil[clus_1d] else clus_1d)
        if sil[clus]>(sil['gtn']+0.1):
            refine_2D(df, clus, dist_sq, nclus)
    return


def refine_2D(df, clus, dist_sq, nclus):
    if nclus==2:
        refine_clusters2_2D(df, clus, dist_sq)
    elif nclus==3:
        refine_clusters3_2D(df, clus, dist_sq)
        df.to_csv('./df_b.csv')
        sys.exit(1)
    else:
        sys.stderr.write("Error:  Number of clusters is not 2 or 3.\n")
        sys.exit(1)
    return
        

def refine_clusters3_2D(df, clus, dist_sq):
    ntot=df.shape[0]
    cc=reorder_clusters_2D(df, clus)
    if cc.iloc[0,:].loc['CN']>cc.iloc[1,:].loc['CN'] and cc.iloc[0,:].loc['CN']>cc.iloc[2,:].loc['CN'] and cc.iloc[0,:].loc['ct']>0.1*ntot:
        mp={cc.index[0]:1, cc.index[1]:2, cc.index[2]:3}
        df.loc[:,'gt_new']=df.loc[:, clus].copy().map(mp)
        df.loc[:,'gt_adj']=df.loc[:, 'gt_new'].copy()
        adjust_mismatches(df, dist_sq, 1, 2)
        adjust_mismatches(df, dist_sq, 2, 3)
        adjust_mismatches(df, dist_sq, 1, 3)
    return

def refine_clusters2_2D(df, clus, dist_sq):
    ntot=df.shape[0]
    cc=reorder_clusters_2D(df, clus)
    if cc.iloc[0,:].loc['CN']>cc.iloc[1,:].loc['CN'] and cc.iloc[0,:].loc['ct']>0.1*ntot:
        mp={cc.index[0]:1, cc.index[1]:2}
        df.loc[:,'gt_new']=df.loc[:, clus].copy().map(mp)
        df.loc[:,'gt_adj']=df.loc[:, 'gt_new'].copy()
        adjust_mismatches(df, dist_sq, 1, 2)
    return

        

def percentile(n):
    def percentile_(x):
        return np.percentile(x,n)
    percentile_.__name__= 'percentile_%s' % n
    return percentile_


#reorder clusters based on upper quartile of AB values
def reorder_clusters_2D(df, lab):    
    gpd=df.loc[:, [lab, 'CN', 'AB']].groupby([lab])
    clusters=gpd.agg({'CN':np.median, 'AB':percentile(75)})
    clusters.loc[:, 'ct']=gpd.size()
    clusters.sort_values(by='AB', inplace=True)
    return clusters

def reorder_clusters_1D(df, lab):    
    gpd=df.loc[:, [lab, 'AB']].groupby([lab])
    clusters=gpd.agg({'AB':percentile(75)})
    clusters.loc[:, 'ct']=gpd.size()
    clusters.sort_values(by='AB', inplace=True)
    return clusters


def load_df(var, exclude, sex):

    test_set = list()

    for s in var.sample_list:
        if s in exclude:
            continue
        cn = var.genotype(s).get_format('CN')
        if (var.chrom == 'X' or var.chrom == 'Y') and sex[s] == 1:
            cn=str(float(cn)*2)
        test_set.append(vcf_rec(var.var_id, s, var.info['SVTYPE'], var.info['AF'],
             var.genotype(s).get_format('GT'),  cn , var.genotype(s).get_format('AB')))

    test_set = pd.DataFrame(data = test_set, columns=vcf_rec._fields)
    return test_set


def run_gt_refine(vcf_in, vcf_out, diag_outfile, gender_file, exclude_file):

    vcf = Vcf()
    header = []
    in_header = True
    sex={}
    
    for line in gender_file:
        v = line.rstrip().split('\t')
        sex[v[0]] = int(v[1])

    exclude = []
    if exclude_file is not None:
        for line in exclude_file:
            exclude.append(line.rstrip())

    outf=open(diag_outfile, 'w', 4096)
    ct=1
    
    bigdf=pd.DataFrame()
    for line in vcf_in:
        if in_header:
            if line[0] == "#":
               header.append(line)
               continue
            else:
                in_header = False
                vcf.add_header(header)
                vcf_out.write(vcf.get_header() + '\n')

        v = line.rstrip().split('\t')
        info = v[7].split(';')
        svtype = None
        for x in info:
            if x.startswith('SVTYPE='):
                svtype = x.split('=')[1]
                break
        # bail if not DEL or DUP prior to reclassification
        if svtype not in ['DEL', 'DUP']:
            vcf_out.write(line)
            continue
        
        var = Variant(v, vcf)
        sys.stderr.write("%s\n" % var.var_id)
        #count positively genotyped samples
        num_pos_samps = 0
        num_total_samps=len(var.sample_list)

        #for s in var.sample_list:
        #    if var.genotype(s).get_format('GT') not in ["./.", "0/0"]:
        #        num_pos_samps += 1
        
        sys.stderr.write("%f\n" % float(var.get_info('AF')))
        if float(var.get_info('AF'))<0.1:
            vcf_out.write(line)
        else:
            df=load_df(var, exclude, sex)
            recdf=recluster(df)
            if ct==1:
                recdf.to_csv(outf, header=True)
                ct += 1
            else:
              recdf.to_csv(outf, header=False)

    vcf_out.close()
    vcf_in.close()
    gender_file.close()
    outf.close()
    if exclude_file is not None:
        exclude_file.close()
    return

            



def add_arguments_to_parser(parser):
    parser.add_argument('-i', '--input', metavar='<VCF>', dest='vcf_in', type=argparse.FileType('r'), default=None, help='VCF input [stdin]')
    parser.add_argument('-o', '--output', metavar='<VCF>', dest='vcf_out', type=argparse.FileType('w'), default=sys.stdout, help='VCF output [stdout]')
    parser.add_argument('-d', '--diag_file', metavar='<STRING>', dest='diag_outfile', type=str, default=None, required=False, help='text file to output method comparisons')
    parser.add_argument('-g', '--gender', metavar='<FILE>', dest='gender', type=argparse.FileType('r'), required=True, default=None, help='tab delimited file of sample genders (male=1, female=2)\nex: SAMPLE_A\t2')
    parser.add_argument('-e', '--exclude', metavar='<FILE>', dest='exclude', type=argparse.FileType('r'), required=False, default=None, help='list of samples to exclude from classification algorithms')
    parser.set_defaults(entry_point=run_from_args)

def description():
    return 'refine genotypes by clustering'

def command_parser():
    parser = argparse.ArgumentParser(description=description())
    add_arguments_to_parser(parser)
    return parser

def run_from_args(args):
    with su.InputStream(args.vcf_in) as stream:
        run_gt_refine(stream, args.vcf_out, args.diag_outfile, args.gender, args.exclude)


if __name__ == '__main__':
    parser = command_parser()
    args=parser.parse_args()
    sys.exit(args.entry_point(args))
