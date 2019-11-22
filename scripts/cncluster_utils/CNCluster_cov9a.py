import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn import cluster
import gaussian_mixture_constr
import CNWindow_cov9
from statsmodels import robust


def min1(xx):
  mm=min(xx)
  if mm==1 and max(xx)>1:
    mm=min([ii for ii in xx if  ii>1])
  return(mm)



class CNClusterExact:
  
  def __init__(self, clus_vars, cn_comp, carriers_comp, verbose, nocl_max=20, nmads=5):
    self.combined_id=str(clus_vars.comp[0])+'_'+str(clus_vars.cluster[0])+'_'+str(clus_vars.dist_cluster[0])
    self.clus_info=clus_vars
    self.cndata=cn_comp
    self.chrom=cn_comp.chr[0]
    self.carriers=carriers_comp
    self.nsamp=np.unique(self.cndata.id).size
    self.ncarriers=self.clus_info.cluster_info_ncarriers[0]
    self.nmads=nmads
    self.verbose=verbose
    self.nocl_max=int(2.5*(np.max(self.cndata['cn'])-np.min(self.cndata['cn']))+2)
    self.subjlist=np.unique(self.cndata.id)
    
  def fit_one_window(self, varstart, varstop, varid, ncarriers):
    cn1=self.cndata[(self.cndata.varstart==varstart) & (self.cndata.varstop==varstop)].copy().reset_index(drop=True)
    win=CNWindow_cov9.CNWindow(self.combined_id, varid, varstart, varstop, cn1.cn,  self.nocl_max, ncarriers, self.verbose)
    fit=win.fit_all_models()
    return fit

  def fit_mixture(self):
    if self.verbose:
      print("fitting mixture")
    fits=[]
    for ii in range(self.clus_info.shape[0]):
      fits.append(self.fit_one_window(self.clus_info.varstart.values[ii], self.clus_info.varstop.values[ii],  self.clus_info.varid.values[ii], self.clus_info.info_ncarriers.values[ii]))
    fits=np.concatenate(fits)
    return self.make_call_mixture(fits)
      

  def make_call_mixture(self, fits):
    fits1=pd.DataFrame(fits, columns=['combined_id', 'varid', 'varstart', 'varstop', 'nocl',
                                      'bic', 'icl', 'sep', 'offset', 'info', 'cn_med', 'cn_mad', 'info_ncarriers', 'dipp'])
    fits1=fits1.astype(dtype={'combined_id': 'string', 'varid': 'string', 'varstart':'int64', 'varstop':'int64',
                              'nocl':'int64', 'bic':'float64', 'icl':'float64', 'sep':'float64', 'offset':'float64', 'info':'string', 
                              'cn_med':'float64',  'cn_mad':'float64', 'info_ncarriers':'int64', 'dipp':'float64'})
    fits1['region']=self.chrom+':'+fits1['varstart'].map(str)+'-'+fits1['varstop'].map(str)
    df=fits1.groupby(['region']).agg({'icl': ['min', 'max']}).reset_index()
    df.columns=['region', 'min_icl', 'max_icl']
    df['thresh1']=df['min_icl']+0.05*(df['max_icl']-df['min_icl'])
    df['thresh2']=df['min_icl']+0.1*(df['max_icl']-df['min_icl'])
    fits2=fits1.merge(df, on=['region'])
    fits3=fits2.loc[fits2.icl<fits2.thresh1].copy()
    ncomp=1
    if fits3.shape[0]>0:
      ncomp=max(fits3.groupby(['region']).agg({'nocl': min1}).reset_index()['nocl'])
    if ncomp>1:
      ret=fits2.loc[(fits2.nocl==ncomp) & (fits2.icl<fits2.thresh2)].sort_values(by=['sep'], ascending=False).head(n=1)
      qual='PASS'
    elif min(fits2['dipp'])<0.05:
      ret=fits2.loc[fits2.nocl==1].sort_values(by=['dipp', 'info_ncarriers']).head(n=1)
      qual='PASS'
    else:
      ret=fits2.loc[fits2.nocl==1].sort_values(by=['info_ncarriers']).head(n=1)
      qual='LOWQUAL'
    ret['qual']=qual
    nvar=self.clus_info.shape[0]
    starts=','.join(self.clus_info['varstart'].map(str))
    stops=','.join(self.clus_info['varstop'].map(str))
    varstart=ret['varstart'].values[0]
    varstop=ret['varstop'].values[0]
    [minstart, maxstart, minstop, maxstop]=[min(self.clus_info['varstart']), max(self.clus_info['varstart']),
                                           min(self.clus_info['varstop']), max(self.clus_info['varstop'])]
    [ptspos, ptsend]=[str(minstart-varstart)+','+str(maxstart-varstart),
                      str(minstop-varstop)+','+str(maxstop-varstop)]
    cn_mad=ret['cn_mad'].values[0]
    cn_med=ret['cn_med'].values[0]
    n_outliers=0
    bic=ret['bic'].values[0]
    icl=ret['icl'].values[0]
    sep=ret['sep'].values[0]
    offset=ret['offset'].values[0]
    info=ret['info'].values[0]+';STARTS='+starts+';STOPS='+stops+';NVAR='+str(nvar)+';CIPOS='+ptspos+';CIEND='+ptsend
    info=info+';CN_MED='+str(cn_med)+';CN_MAD='+str(cn_mad)+';CN_CARRIER_MED=.;N_OUTLIERS=0'
    info=info+';SEP='+str(sep)+';OFFSET='+str(offset)+';ICL='+str(icl)+';BIC='+str(bic)+';RARE=False'
    ret['info']=info
    ret=ret[['combined_id', 'varstart', 'varstop', 'region', 'varid', 'nocl', 'info', 'qual']]
    return [qual, ret]
        
    
  def fit_generic(self, outf1):

    [qual, fit]=self.fit_mixture()
    fit_rare=None
    if  qual=='LOWQUAL' and (self.ncarriers<0.0025*self.nsamp or self.ncarriers<10):
      fit_rare=self.check_outliers_all()
      if fit_rare is not None:
        fit=fit_rare
    fit.to_csv('fit2.csv')
    cn=self.cndata[['varid', 'id', 'cn']].copy().merge(fit, on='varid')
    if sum(cn.id!=self.subjlist)==0:
      cn1='\t'.join(cn['cn'].map(str))
    outstr=self.chrom+'\t'+str(fit['varstart'].values[0])+'\t'+str(fit['varstop'].values[0])+'\t'+str(fit['nocl'].values[0])+'\t'+str(fit['qual'].values[0])+'\t'+str(fit['info'].values[0])+'\t'+cn1+'\n'
    fit.to_csv('fit1.csv')
    outf1.write(outstr)
    return 


  def check_outliers_all(self):
      if self.carriers.shape[0]>0:
        dd=self.cndata.copy().rename(columns={ 'info_ncarriers':'info_ncarriers_var'})
        cn1=dd.merge(self.carriers,  on=['varid', 'id', 'comp'], how='left')
        cn1['carrier']=np.where(cn1['info_ncarriers'].isnull(), 'non', 'carrier')
        #number of carriers of any variant in cluster
        ncar=cn1.loc[cn1.carrier=="carrier"].shape[0]
        if ncar > 0:
          cn1_ag=cn1.groupby(['varid', 'varstart', 'varstop', 'carrier'])['cn'].aggregate({'cn_med': np.median, 'cn_mad':robust.stand_mad}).reset_index()
          ag_carriers=cn1_ag.loc[cn1_ag.carrier=="carrier", ['varid', 'cn_med']].copy().rename(columns={'cn_med':'carrier_med'})
          ag_non=cn1_ag.loc[cn1_ag.carrier=="non", ['varid', 'varstart', 'varstop', 'cn_med', 'cn_mad']].copy()
          ag=ag_non.merge(ag_carriers, on=['varid'])
          ag['ll']=ag['cn_med']-self.nmads*ag['cn_mad']
          ag['ul']=ag['cn_med']+self.nmads*ag['cn_mad']
          ag['dist']=(ag['cn_med']-ag['carrier_med'])/ag['cn_mad']
          ag['ncarriers']=ncar
          if ag.loc[np.abs(ag.dist)>self.nmads].shape[0]>0:
            eps=1e-6
            dd2=dd[['id', 'cn', 'varid']].merge(ag, on='varid')
            dd2['outlier_up']=np.where(dd2['cn']>dd2['ul']-eps, "outlier", "not")
            dd2['outlier_down']=np.where(dd2['cn']<dd2['ll']+eps, "outlier", "not")
            dd2['is_outlier']=np.where(dd2['dist']<0, dd2['outlier_up'], dd2['outlier_down'])
            cts=dd2.groupby(['varid', 'is_outlier'])['id'].count().reset_index().rename(columns={'id': 'n_outliers'})
            ag=ag.merge(cts.loc[cts.is_outlier=='outlier'].copy(), on='varid')
            if ag.shape[0]>0 and np.min(ag.n_outliers)<0.01*self.nsamp:
               return self.make_call_rare(ag) 

  def make_call_rare(self, fits):

    fits['abs_dist']=abs(fits['dist'])
    fits=fits.sort_values(by=['abs_dist'], ascending=False).head(n=1)    
    fits['combined_id']=self.combined_id
    fits['region']=self.chrom+':'+fits['varstart'].map(str)+'-'+fits['varstop'].map(str)
    fits['nocl']=1
    fits['qual']='PASS'
    
    varstart=fits['varstart'].values[0]
    varstop=fits['varstop'].values[0]
    nvar=self.clus_info.shape[0]
    starts=','.join(self.clus_info['varstart'].map(str))
    stops=','.join(self.clus_info['varstop'].map(str))
    [minstart, maxstart, minstop, maxstop]=[min(self.clus_info['varstart']), max(self.clus_info['varstart']),
                                            min(self.clus_info['varstop']), max(self.clus_info['varstop'])]
    [ptspos, ptsend]=[str(minstart-varstart)+','+str(maxstart-varstart),
                      str(minstop-varstop)+','+str(maxstop-varstop)]
    cn_mad=fits['cn_mad'].values[0]
    cn_med=fits['cn_med'].values[0]
    carrier_med=fits['carrier_med'].values[0]
    n_outliers=fits['n_outliers'].values[0]
    info='STARTS='+starts+';STOPS='+stops+';NVAR='+str(nvar)+';CIPOS='+ptspos+';CIEND='+ptsend
    info=info+';CN_MED='+str(cn_med)+';CN_MAD='+str(cn_mad)+';CN_CARRIER_MED='+str(carrier_med)+';N_OUTLIERS='+str(n_outliers)
    info=info+';SEP=0;OFFSET=0;ICL=0;BIC=0;RARE=True'
    fits['info']=info
    fit=fits[['combined_id', 'varstart', 'varstop', 'region', 'varid', 'nocl', 'info', 'qual']]
    return fits
