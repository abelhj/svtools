import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn import cluster
import gaussian_mixture_constr
import CNWindow1a
from statsmodels import robust

class CNClusterExact:
  
  def __init__(self, clus_vars, cn_comp, carriers_comp, nocl_min=1, nocl_max=20, nmads=5):
    self.comp_id=clus_vars.comp[0]
    self.clus_info=clus_vars
    self.cndata=cn_comp
    self.clus_id=clus_vars.cluster[0]
    self.dist_clus_id=clus_vars.dist_cluster[0]
    self.nocl_min=nocl_min
    self.nocl_max=nocl_max
    self.carriers=carriers_comp
    self.nsamp=np.unique(self.cndata.id).size
    self.ncarriers=self.clus_info.cluster_info_ncarriers[0]
    self.nmads=nmads

  def get_chunk_data(self, chunkstart, chunkstop):
    cn=self.cndata[(self.cndata.varstart==chunkstart) & (self.cndata.varstop==chunkstop)].reset_index()
    cn['chunkstart']=chunkstart
    cn['chunkstop']=chunkstop
    cn['clus_id']=self.clus_id
    cn['dist_clus_id']=self.dist_clus_id
    cn.rename(columns={'cn1': 'cn'}, inplace=True)
    return cn

  def fit_one_window(self, chunkstart, chunkstop, ncarriers):
    cn1=self.get_chunk_data(chunkstart, chunkstop)
    win=CNWindow1a.CNWindow(self.comp_id, self.clus_id, self.dist_clus_id, chunkstart, chunkstop, cn1.cn, self.nocl_min, self.nocl_max)
    fit=win.fit_all_models(ncarriers)
    return fit

  def fit_mixture(self):
    print("fitting mixture")
    fits=[]
    data=[]
    for ii in range(self.clus_info.shape[0]):
      fit=self.fit_one_window(self.clus_info.varstart.values[ii], self.clus_info.varstop.values[ii],  self.clus_info.info_ncarriers.values[ii])
      fits.append(fit)
    fits=np.concatenate(fits)
    self.make_call_mixture(fits)


  def make_call_mixture(self, fits):
    fits1=pd.DataFrame(fits, columns=['comp', 'clus_id', 'dist_clus_id', 'chunkstart', 'chunkstop', 'nocl',
                                'bic', 'mm', 'kk', 'cn_med', 'cn_mad', 'info_ncarriers', 'nocl_sel'])
    fits1=fits1.astype(dtype={'comp': 'int64', 'clus_id':'int64', 'dist_clus_id':'int64', 'chunkstart':'int64', 'chunkstop':'int64',
                             'nocl':'int64', 'bic':'float64', 'mm':'float64', 'kk':'float64', 'cn_med':'float64',  'cn_mad':'float64',
                             'info_ncarriers':'int64', 'nocl_sel':'int64'})
    fits1['is_rare']=False
    fits1=fits1.loc[fits1.nocl==fits1.nocl_sel] #contains one call per variant
    nocl_max=np.max(fits1['nocl'])
    fits1['corr']=1.0*fits1['nocl']/nocl_max
    fits1['mm_corr']=fits1['mm']*fits1['corr']
    if nocl_max>1:
      fits1=self.get_cis(fits1, 'mm_corr')
    else:
      fits1=self.get_cis(fits1, 'info_ncarriers')
    return fits1

  def make_call_rare(self, fits):
    fits['is_rare']=True
    fits['mm']=0
    fits['kk']=0
    fits['bic']=0
    fits['nocl']=0
    fits=fits[['comp', 'clus_id', 'dist_clus_id', 'chunkstart', 'chunkstop', 'nocl',
               'carrier_med', 'cn_med', 'cn_mad', 'dist', 'n_outliers', 'ncarriers',
               'info_ncarriers_var', 'is_rare', 'mm', 'kk', 'bic']]
                                                            
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
      fits["is_winner"]="winner"
    else:
      startpt=np.min(fits.loc[fits[score]==np.max(fits[score]), 'chunkstart'])
      stoppt=np.max(fits.loc[fits[score]==np.max(fits[score]), 'chunkstop'])
      starts=fits[['chunkstart', score]].rename(columns={'chunkstart': 'pos'}).reset_index(drop=True)
      starts['end']='start'
      stops=fits[['chunkstop', score]].rename(columns={'chunkstop': 'pos'}).reset_index(drop=True)
      stops['end']='stop'
      bks1=pd.concat([starts, stops], axis=0)
      bks1['endpt']=np.where(bks1.end=='start', startpt, stoppt)
      bks1['diff']=bks1['pos']-bks1['endpt']
      bks1[score]=1.0*bks1[score]

      bks1_ag=bks1.groupby(['pos', 'end', 'diff'])[score].aggregate([np.sum]).reset_index().sort_values(by=['pos'])
      bks1_ag_left1=bks1_ag.loc[(bks1_ag.diff<=0) & (bks1_ag.end=='start')]
      bks1_ag_left1['pr']=np.cumsum( bks1_ag_left1['sum'])/np.sum( bks1_ag_left1['sum'])
      bks1_ag_left2=bks1_ag.loc[(bks1_ag.diff>=0) & (bks1_ag.end=='start')]
      bks1_ag_left2['pr']=np.cumsum(bks1_ag_left2['sum'].values[::-1])[::-1]/np.sum(bks1_ag_left2['sum'])
      bks1_ag_right1=bks1_ag.loc[(bks1_ag.diff<=0) & (bks1_ag.end=='stop')]
      bks1_ag_right1['pr']=np.cumsum( bks1_ag_right1['sum'])/np.sum(bks1_ag_right1['sum'])
      bks1_ag_right2=bks1_ag.loc[(bks1_ag.diff>=0) & (bks1_ag.end=='stop')]
      bks1_ag_right2['pr']=np.cumsum(bks1_ag_right2['sum'].values[::-1])[::-1]/np.sum(bks1_ag_right2['sum'])
      bks2=pd.concat([bks1_ag_left1, bks1_ag_left2, bks1_ag_right1, bks1_ag_right2], axis=0).drop_duplicates()

      fits['ptsend']=','.join( map(str, bks2.loc[bks2.end=='stop', 'diff'].values))
      fits['ptspos']=','.join( map(str, bks2.loc[bks2.end=='start', 'diff'].values))
      fits['prend']=','.join( map(str, bks2.loc[bks2.end=='stop', 'pr'].values))
      fits['prpos']=','.join( map(str, bks2.loc[bks2.end=='start', 'pr'].values))
      fits['is_winner']=np.where((fits['chunkstart']==startpt) & (fits['chunkstop']==stoppt), 'winner', 'not_winner')
    return fits
    

  def fit_generic(self, outf1, outf2):
    #print(str(self.ncarriers)+"\t"+str(self.nsamp))
    if self.ncarriers>0.0025*self.nsamp and self.ncarriers>=10:
      fit=self.fit_mixture()
    else:   
      print("try rare")
      fit=self.check_outliers_all()
    #np.savetxt(outf1, fit, delimiter="\t", fmt="%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f")
    return fit

  def check_outliers_all(self):
    dd=[]
    fits=[]
    for ii in range(self.clus_info.shape[0]):
      cn1=self.get_chunk_data(self.clus_info.varstart.values[ii], self.clus_info.varstop.values[ii])
      cn1['varid']=self.clus_info.varid.values[ii]
      cn1['info_ncarriers_var']=self.clus_info.info_ncarriers.values[ii]
      dd.append(cn1)
    dd=pd.concat(dd)
    #dd is copy number data for all variants in cluster
    cn1=dd.merge(self.carriers,  on=['varid', 'id', 'comp'], how='left')
    cn1['carrier']=np.where(cn1['info_ncarriers'].isnull(), 'non', 'carrier')
    #number of carriers of any variant in cluster
    ncar=cn1.loc[cn1.carrier=="carrier"].shape[0]
    if ncar > 0:
      cn1_ag=cn1.groupby(['varid', 'carrier', 'clus_id', 'dist_clus_id', 'comp', 'chunkstart', 'chunkstop', 'info_ncarriers_var'])['cn'].agg({'cn_med': np.median, 'cn_mad':robust.stand_mad}).reset_index()
      ag_carriers=cn1_ag.loc[cn1_ag.carrier=="carrier"][['varid', 'clus_id', 'dist_clus_id', 'comp', 'cn_med', 'chunkstart', 'chunkstop', 'info_ncarriers_var']].rename(columns={'cn_med':'carrier_med'})
      ag_non=cn1_ag.loc[cn1_ag.carrier=="non"]   
      ag=ag_non.merge(ag_carriers, on=['varid', 'clus_id', 'dist_clus_id', 'comp', 'chunkstart', 'chunkstop', 'info_ncarriers_var'])
      ag['ll']=ag['cn_med']-self.nmads*ag['cn_mad']
      ag['ul']=ag['cn_med']+self.nmads*ag['cn_mad']
      ag['dist']=(ag['cn_med']-ag['carrier_med'])/ag['cn_mad']
      ag['n_outliers']=0
      ag['ncarriers']=ncar
      #ag1=ag.loc[(ag.dist>self.nmads) | (ag.dist < -1*self.nmads) ]
      #ag.to_csv('ag.csv')
      #exit(1)
      ag1=ag.loc[np.abs(ag.dist)>self.nmads]
      rare=False
      if ag1.shape[0]>0:
        ag.to_csv('ag.csv')
        dd.to_csv('dd.csv')
        exit(1)
        for varid in ag1['varid'].values:
          cn2=dd.loc[dd.varid == varid]
          cn2=cn2.merge(ag1[['varid', 'll', 'ul', 'dist']], on=['varid'])
          if cn2['dist'].values[0]<0:
            cn2=cn2.loc[cn2.cn>cn2.ul]
          else:
            cn2=cn2.loc[cn2.cn<cn2.ll]
          n_outlier=cn2.shape[0]
          ag.loc[ag.varid==varid, 'n_outliers']=n_outlier
          if n_outlier<0.01*self.nsamp:
            rare=True
            print("rare variant")
      if rare:
        self.make_call_rare(ag)
        return 
    return self.fit_mixture()
    
    
