import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn import cluster
import gaussian_mixture_constr
import CNWindow_cov7
from statsmodels import robust

class CNClusterExact:
  
  def __init__(self, clus_vars, cn_comp, carriers_comp, verbose, nocl_max=20, nmads=5):
    self.comp_id=clus_vars.comp[0]
    self.clus_info=clus_vars
    self.cndata=cn_comp
    self.clus_id=clus_vars.cluster[0]
    self.chrom=cn_comp.chr[0]
    self.dist_clus_id=clus_vars.dist_cluster[0]
    self.nocl_max=nocl_max
    self.carriers=carriers_comp
    self.nsamp=np.unique(self.cndata.id).size
    self.ncarriers=self.clus_info.cluster_info_ncarriers[0]
    self.nmads=nmads
    self.verbose=verbose

  def get_chunk_data(self, chunkstart, chunkstop):
    cn=self.cndata[(self.cndata.varstart==chunkstart) & (self.cndata.varstop==chunkstop)].copy().reset_index(drop=True)
    cn['chunkstart']=chunkstart
    cn['chunkstop']=chunkstop
    cn['clus_id']=self.clus_id
    cn['dist_clus_id']=self.dist_clus_id
    return cn

  def fit_one_window(self, chunkstart, chunkstop, ncarriers):
    cn1=self.get_chunk_data(chunkstart, chunkstop)
    win=CNWindow_cov7.CNWindow(self.comp_id, self.clus_id, self.dist_clus_id, chunkstart, chunkstop, cn1.cn,  self.nocl_max, ncarriers, self.verbose)
    fit=win.fit_all_models()
    return fit

  def fit_mixture(self):
    if self.verbose:
      print("fitting mixture")
    fits=[]
    data=[]
    
    for ii in range(self.clus_info.shape[0]):
      fit=self.fit_one_window(self.clus_info.varstart.values[ii], self.clus_info.varstop.values[ii],  self.clus_info.info_ncarriers.values[ii])
      fits.append(fit)
    fits=np.concatenate(fits)
    return self.make_call_mixture(fits)


  def make_call_mixture(self, fits):
    fits1=pd.DataFrame(fits, columns=['comp', 'clus_id', 'dist_clus_id', 'chunkstart', 'chunkstop', 'nocl',
                                      'bic', 'icl', 'mm', 'kk', 'info', 'cn_med', 'cn_mad', 'info_ncarriers', 'nocl_sel', 'dipp'])
    fits1=fits1.astype(dtype={'comp': 'int64', 'clus_id':'int64', 'dist_clus_id':'int64', 'chunkstart':'int64', 'chunkstop':'int64',
                              'nocl':'int64', 'bic':'float64', 'icl':'float64', 'mm':'float64', 'kk':'float64', 'info':'string', 
                              'cn_med':'float64',  'cn_mad':'float64', 'info_ncarriers':'int64', 'nocl_sel':'int64', 'dipp':'float64'})
    fits1['is_rare']=False
    fits1['n_outliers']=0
    fits1['chrom']=self.chrom
    fits1=fits1[fits1.nocl==fits1.nocl_sel].copy().reset_index(drop=True) #contains one call per variant
    fits1['negicl']=-1.0*fits1['icl']
    fits1.loc[np.isnan(fits1.icl), ['negicl']]=0.5*np.min(fits1.negicl)
    fits1['dist']=fits1['mm']
    fits1=fits1[['comp', 'clus_id', 'dist_clus_id', 'chrom', 'chunkstart', 'chunkstop',
                        'nocl', 'bic', 'icl', 'mm', 'kk', 'info', 'cn_med', 'cn_mad', 'info_ncarriers',
                        'is_rare', 'negicl', 'dist', 'dipp', 'n_outliers']].copy()
    if np.max(fits1['nocl'])>1:
      fits1=self.get_cis(fits1, 'negicl')
    else:
      fits1=self.get_cis(fits1, 'info_ncarriers')
    return fits1

  def make_call_rare(self, fits):
    fits['is_rare']=True
    fits['mm']=0
    fits['kk']=0
    fits['info']='.'
    fits['bic']=0
    fits['icl']=0
    fits['nocl']=0
    fits['negicl']=1.0
    fits['dipp']=-1.0
    fits['chrom']=self.chrom
    fits['freq']=0.0
    fits.rename(columns={'info_ncarriers_var':'info_ncarriers'}, inplace=True)
    fits=fits[['comp', 'clus_id', 'dist_clus_id', 'chrom', 'chunkstart', 'chunkstop',
                        'nocl', 'bic', 'icl', 'mm', 'kk', 'info', 'freq', 'cn_med', 'cn_mad', 'info_ncarriers',
                        'is_rare', 'negbic', 'dist', 'dipp', 'n_outliers']].copy()
    fits=self.get_cis(fits, 'dist')
    return fits



  def get_cis(self, fits, score):

    fits['nvar']=fits.shape[0]
    fits['score']=score
    if fits.shape[0]==1:
      fits['ptspos']='0'
      fits['ptsend']='0'
      fits['is_winner']='winner'
    else:
      startpt=np.min(fits.loc[fits[score]==np.max(fits[score]), 'chunkstart'])
      stoppt=np.max(fits.loc[fits[score]==np.max(fits[score]), 'chunkstop'])
      [minstart, maxstart]=[np.min(fits['chunkstart']), np.max(fits['chunkstart'])]
      [minstop, maxstop]=[np.min(fits['chunkstop']), np.max(fits['chunkstop'])]
      fits['ptspos']=str(minstart-startpt)+','+str(maxstart-startpt)
      fits['ptsend']=str(minstop-stoppt)+','+str(maxstop-stoppt)
      fits['is_winner']=np.where((fits['chunkstart']==startpt) & (fits['chunkstop']==stoppt), 'winner', 'not_winner')
    return fits

  
    

  def fit_generic(self, outf1, outf2, outf3, outf4):

    fit=self.fit_mixture()
    fit.to_csv(outf1, sep='\t', mode='a', float_format='%.3f', header=False, index=False, na_rep='NA')
    fit=fit.loc[fit.is_winner=="winner"]
    fit.to_csv(outf2, sep='\t', mode='a', float_format='%.3f', header=False, index=False, na_rep='NA')
    fit_rare=None
    if  self.ncarriers<0.0025*self.nsamp or  self.ncarriers<10:
      fit_rare=self.check_outliers_all()
      if fit_rare is not None:
        fit_rare.to_csv(outf3, sep='\t', mode='a', float_format='%.3f', header=False, index=False, na_rep='NA')
        fit_rare=fit_rare.loc[fit_rare.is_winner=="winner"]
        fit_rare.to_csv(outf3, sep='\t', mode='a', float_format='%.3f', header=False, index=False, na_rep='NA')
    return 


  def check_outliers_all(self):
      if self.carriers.shape[0]>0:

        dd=self.cndata.drop(['svlen'], axis=1).copy()
        dd.rename(columns={ 'varstart':'chunkstart', 'varstop':'chunkstop', 'info_ncarriers':'info_ncarriers_var'}, inplace=True)
        cn1=dd.merge(self.carriers,  on=['varid', 'id', 'comp'], how='left')
        cn1['carrier']=np.where(cn1['info_ncarriers'].isnull(), 'non', 'carrier')
        #number of carriers of any variant in cluster
        ncar=cn1.loc[cn1.carrier=="carrier"].shape[0]
        if ncar > 0:
          cn1_ag=cn1.groupby(['varid', 'carrier', 'comp', 'chunkstart', 'chunkstop', 'info_ncarriers_var'])['cn'].agg({'cn_med': np.median, 'cn_mad':robust.stand_mad}).reset_index()
          ag_carriers=cn1_ag.loc[cn1_ag.carrier=="carrier", ['varid', 'clus_id', 'comp', 'cn_med', 'chunkstart', 'chunkstop', 'info_ncarriers_var']].copy()
          ag_carriers.rename(columns={'cn_med':'carrier_med'}, inplace=True)
          ag_non=cn1_ag.loc[cn1_ag.carrier=="non"].copy()
          ag=ag_non.merge(ag_carriers, on=['varid', 'comp', 'chunkstart', 'chunkstop', 'info_ncarriers_var'])
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
            cts=dd2.groupby(['varid', 'is_outlier'])['comp'].count().reset_index().rename(columns={'comp':'n_outliers'})
            ag=ag.merge(cts.loc[cts.is_outlier=='outlier',['varid', 'n_outliers']].copy(), on='varid')
            ag['clus_id']=self.clus_info.cluster.values[0]
            ag['dist_clus_id']=self.clus_info.dist_cluster.values[0]
            if np.min(ag.n_outliers)<0.01*self.nsamp:
               return self.make_call_rare(ag) 
