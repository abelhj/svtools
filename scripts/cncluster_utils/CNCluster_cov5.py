import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn import cluster
import gaussian_mixture_constr
import CNWindow_cov4
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
    win=CNWindow_cov4.CNWindow(self.comp_id, self.clus_id, self.dist_clus_id, chunkstart, chunkstop, cn1.cn,  self.nocl_max, ncarriers, self.verbose)
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
                                      'bic', 'mm', 'kk', 'cov', 'wts', 'freq', 'cn_med', 'cn_mad', 'info_ncarriers', 'nocl_sel', 'dipp'])
    fits1=fits1.astype(dtype={'comp': 'int64', 'clus_id':'int64', 'dist_clus_id':'int64', 'chunkstart':'int64', 'chunkstop':'int64',
                              'nocl':'int64', 'bic':'float64', 'mm':'float64', 'kk':'float64', 'cov':'string', 'wts':'string', 'freq':'float64',
                              'cn_med':'float64',  'cn_mad':'float64', 'info_ncarriers':'int64', 'nocl_sel':'int64', 'dipp':'float64'})
    fits1['is_rare']=False
    fits1['n_outliers']=0
    fits1['chrom']=self.chrom
    fits1=fits1[fits1.nocl==fits1.nocl_sel].copy().reset_index(drop=True) #contains one call per variant
    fits1['negbic']=-1.0*fits1['bic']
    fits1.loc[np.isnan(fits1.bic), ['negbic']]=0.5*np.min(fits1.negbic)
    fits1['dist']=fits1['mm']
    fits1=fits1[['comp', 'clus_id', 'dist_clus_id', 'chrom', 'chunkstart', 'chunkstop',
                        'nocl', 'bic', 'mm', 'kk', 'cov', 'wts', 'freq', 'cn_med', 'cn_mad', 'info_ncarriers',
                        'is_rare', 'negbic', 'dist', 'dipp', 'n_outliers']].copy()
    if np.max(fits1['nocl'])>1:
      fits1=self.get_cis(fits1, 'negbic')
    else:
      fits1=self.get_cis(fits1, 'info_ncarriers')
    return fits1

  def make_call_rare(self, fits):
    fits['is_rare']=True
    fits['mm']=0
    fits['kk']=0
    fits['cov']='.'
    fits['wts']='.'
    fits['bic']=0
    fits['nocl']=0
    fits['negbic']=1.0
    fits['dipp']=-1.0
    fits['chrom']=self.chrom
    fits['freq']=0.0
    fits.rename(columns={'info_ncarriers_var':'info_ncarriers'}, inplace=True)
    fits=fits[['comp', 'clus_id', 'dist_clus_id', 'chrom', 'chunkstart', 'chunkstop',
                        'nocl', 'bic', 'mm', 'kk', 'cov', 'wts', 'freq', 'cn_med', 'cn_mad', 'info_ncarriers',
                        'is_rare', 'negbic', 'dist', 'dipp', 'n_outliers']].copy()
    fits=self.get_cis(fits, 'dist')
    return fits

  
  def get_cis(self, fits, score):

    fits['nvar']=fits.shape[0]
    fits['score']=score

    if fits.shape[0]==1:
      fits['ptspos']="0"
      fits['ptsend']="0"
      fits['prpos']="1"
      fits['prend']="1"
      fits['is_winner']="winner"
    else:
      startpt=np.min(fits.loc[fits[score]==np.max(fits[score]), 'chunkstart'])
      stoppt=np.max(fits.loc[fits[score]==np.max(fits[score]), 'chunkstop'])
      starts=fits[['chunkstart', score]].rename(columns={'chunkstart': 'pos'}).copy().reset_index(drop=True)
      starts['end']='start'
      stops=fits[['chunkstop', score]].rename(columns={'chunkstop': 'pos'}).copy().reset_index(drop=True)
      stops['end']='stop'
      bks1=pd.concat([starts, stops], axis=0)
      bks1['endpt']=np.where(bks1.end=='start', startpt, stoppt)
      bks1['diff']=bks1['pos']-bks1['endpt']
      bks1[score]=1.0*bks1[score]

      bks1_ag=bks1.groupby(['pos', 'end', 'diff'])[score].aggregate([np.sum]).reset_index().sort_values(by=['pos'])
      bks1_ag_left1=bks1_ag[(bks1_ag['diff']<=0) & (bks1_ag.end=='start')].copy()
      bks1_ag_left1['pr']=np.cumsum( bks1_ag_left1['sum'])/np.sum( bks1_ag_left1['sum'])
      bks1_ag_left2=bks1_ag[(bks1_ag['diff']>=0) & (bks1_ag.end=='start')].copy()
      bks1_ag_left2['pr']=np.cumsum(bks1_ag_left2['sum'].values[::-1])[::-1]/np.sum(bks1_ag_left2['sum'])
      bks1_ag_right1=bks1_ag[(bks1_ag['diff']<=0) & (bks1_ag.end=='stop')].copy()
      bks1_ag_right1['pr']=np.cumsum( bks1_ag_right1['sum'])/np.sum(bks1_ag_right1['sum'])
      bks1_ag_right2=bks1_ag[(bks1_ag['diff']>=0) & (bks1_ag.end=='stop')].copy()
      bks1_ag_right2['pr']=np.cumsum(bks1_ag_right2['sum'].values[::-1])[::-1]/np.sum(bks1_ag_right2['sum'])
      bks2=pd.concat([bks1_ag_left1, bks1_ag_left2, bks1_ag_right1, bks1_ag_right2], axis=0).drop_duplicates()

      fits['ptspos']=','.join( map(str, bks2.loc[bks2.end=='start', 'diff'].values))
      fits['ptsend']=','.join( map(str, bks2.loc[bks2.end=='stop', 'diff'].values))
      fits['prpos']=','.join( map(str, bks2.loc[bks2.end=='start', 'pr'].values))
      fits['prend']=','.join( map(str, bks2.loc[bks2.end=='stop', 'pr'].values))
      fits['is_winner']=np.where((fits['chunkstart']==startpt) & (fits['chunkstop']==stoppt), 'winner', 'not_winner')
    return fits
    

  def fit_generic(self):
    fit=self.fit_mixture()
    fit_rare=None
    if  self.ncarriers<0.0025*self.nsamp or  self.ncarriers<10:
      fit_rare=self.check_outliers_all()
    return [fit, fit_rare]


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
