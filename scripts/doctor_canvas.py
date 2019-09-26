import argparse, sys, StringIO
import os
import numpy as np
import pysam

ar=os.path.dirname(os.path.realpath(__file__)).split('/')
svtpath='/'.join(ar[0:(len(ar)-1)])
sys.path.insert(1, svtpath)
from svtools.vcf.file import Vcf
from svtools.vcf.variant import Variant
from collections import namedtuple
import svtools.utils as su


def add_arguments_to_parser(parser):
    parser.add_argument('-i', '--vcf', metavar='<VCF>', dest='manta_vcf', help="manta input vcf")
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('-s', '--slop', dest='slop',  default=0,  required=False, help='padding to either side')
    parser.add_argument('-m', '--max_ins', dest='max_ins', default=1000, type=int, required=False, help='maximum insert size') 

def command_parser():
    parser = argparse.ArgumentParser(description="fix up manta vcfs for lsort and lmerge")
    add_arguments_to_parser(parser)
    return parser

def convert_variant(v, max_ins, chrdict):
    set_read_counts(v)
    set_cis_prs(v)
    if v.get_info('SVTYPE')=='CNV':
        convert_cnv(v)

def split_ci(ci):
     return[int(ci.split(',')[0]),  int(ci.split(',')[1])]

def uniform_pr(length):
    pr=np.ones(length, dtype='float64')/length
    pr1=','.join( map(str, pr))
    return pr1

def set_read_counts(var):
    sample=var.sample_list[0]
    gt=var.genotype(sample)
    pe=0
    sr=0
    if 'PR' in var.format_dict:
        pe=int(gt.get_format('PR').split(',')[1])
    if 'SR' in var.format_dict:
        sr=int(gt.get_format('SR').split(',')[1])
    var.info['PE']=pe
    var.info['SR']=sr
    var.info['SU']=pe+sr

def set_cis_prs(v):
    imprec=False
    cipos='0,0'
    ciend='0,0'
    prpos=1.0
    prend=1.0
    if 'CIPOS' in v.info:
        cipos=v.get_info('CIPOS')
        [start, stop]=split_ci(cipos)
        prpos=uniform_pr(stop-start+1)
        imprec=True
    if 'CIEND' in v.info:
        ciend=v.get_info('CIEND')
        [start, stop]=split_ci(ciend)
        prend=uniform_pr(stop-start+1)
        imprec=True
    v.info['CIPOS']=cipos
    v.info['CIEND']=ciend
    v.info['CIPOS95']=cipos
    v.info['CIEND95']=ciend
    v.info['PRPOS']=prpos
    v.info['PREND']=prend
    v.set_info('IMPRECISE', imprec)
    
def convert_cnv(var):
    var.alt='<DUP>'
    var.info['SVTYPE']='DUP'
    var.info['STRANDS']='-+:2'
    var.ref='N'

def run_from_args(args):

  vcf = Vcf()
  vcf_out=sys.stdout
  in_header = True
  header_lines = list()
  chrdict = {}
  chrnum=1
  with su.InputStream(args.manta_vcf) as input_stream:
    for line in input_stream:
      if in_header:
        header_lines.append(line)
        if line[0:12] == '##contig=<ID':
          chrom=line.replace(',', '=').split('=')[2]
          chrdict[chrom]=chrnum
          chrnum=chrnum+1
        if line[0:6] == '#CHROM':
          in_header=False
          vcf.add_header(header_lines)
          vcf.add_info('PRPOS', '1', 'String', 'Breakpoint probability dist')
          vcf.add_info('PREND', '1', 'String', 'Breakpoint probability dist')
          vcf.add_info('STRANDS', '.', 'String', 'Strand orientation of the adjacency in BEDPE format (DEL:+-, DUP:-+, INV:++/--')
          vcf.add_info('SU', '.', 'Integer', 'Number of pieces of evidence supporting the variant across all samples')
          vcf.add_info('PE', '.', 'Integer', 'Number of paired-end reads supporting the variant across all samples')
          vcf.add_info('SR', '.', 'Integer', 'Number of split reads supporting the variant across all samples')
          vcf.add_info('INSLEN_ORIG', '.', 'Integer', 'Original insertion length')
          vcf.add_info('CIPOS95', '2', 'Integer', 'Confidence interval (95%) around POS for imprecise variants')
          vcf.add_info('CIEND95', '2', 'Integer', 'Confidence interval (95%) around END for imprecise variants')
          vcf.add_info('SECONDARY', '0', 'Flag', 'Secondary breakend in a multi-line variant')
          vcf.add_info('IMPRECISE', '0', 'Flag', 'Imprecise structural variation')
          vcf_out.write(vcf.get_header()+'\n')
      else:
        v = Variant(line.rstrip().split('\t'), vcf)
        if v.alt !=".":
          if v.get_info('SVTYPE')=='CNV':
            convert_variant(v, args.max_ins, chrdict)
            vcf_out.write(v.get_var_string()+"\n")
          

parser=command_parser()
args=parser.parse_args()
run_from_args(args)
