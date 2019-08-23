import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.stats import binom, norm, gamma, uniform, dirichlet, shapiro
import gaussian_mixture_constr
from statsmodels import robust
import sys
sys.path.append('/gscmnt/gc2802/halllab/abelhj/svtools/scripts/cncluster_utils')
import dip


class CNWindow(object):

  def __init__(self, comp_id, clus_id, clus_dist_id, start, stop, cndata, nocl_max, info_carriers, verbose):
    self.data=cndata
    self.nocl_max=nocl_max
    self.clus_id=clus_id
    self.clus_dist_id=clus_dist_id
    self.comp_id=comp_id
    self.start=start
    self.stop=stop
    self.nsamp=self.data.size
    self.procdata=np.empty(0)
    self.info_carriers=info_carriers
    self.cn_med=np.median(cndata)
    self.cn_mad=robust.stand_mad(cndata)
    self.verbose=verbose
    
  def add_noise(self):
    cn2=np.sort(self.data)
    cn2=cn2+norm.rvs(loc=0, scale=0.005, size=cn2.size)
    self.procdata=cn2.reshape(-1, 1)

  def fit_all_models(self):

    fits=[]
    dipp=dip.diptst1(self.data)[1]
    nocl_min=1
    if (dipp>0.9) and (self.info_carriers < 0.01*self.nsamp):
      fits.append(self.fit_no_model())
    else:
      self.add_noise()
      bic_col=6
      mm=2.5*(np.max(self.procdata)-np.min(self.procdata))+2
      if mm>self.nocl_max:
        mm=self.nocl_max
      res=self.fit_one_model(1)
      fits.append(res)
      bic=res[bic_col]
      bic_min=bic
      nocl=2
      if self.verbose:
        print(str(self.start)+"\t"+str(self.stop)+"\t"+str(nocl)+"\t"+str(bic)+"\t"+str(bic_min)+"\t"+str(dipp))
      while (nocl<mm): 
        res=self.fit_one_model(nocl)
        fits.append(res)
        bic=res[bic_col]
        if bic<bic_min:
          bic_min=bic
          nocl_min=nocl
        if self.verbose:
          print(str(self.start)+"\t"+str(self.stop)+"\t"+str(nocl)+"\t"+str(bic)+"\t"+str(bic_min)+"\t"+str(dipp))
        #save some model fitting
        if nocl_min<=2 and nocl==4:  
          break
        elif nocl_min>=3 and nocl>=2*nocl_min:
          break
        nocl=nocl+1
        
    fits=np.vstack(fits)
    fits=np.hstack([fits, np.empty([fits.shape[0], 2], dtype='int32'), np.empty([fits.shape[0], 1], dtype='float64')])
    fits[:, 14]=self.info_carriers
    fits[:,15]=nocl_min
    fits[:,16]=dipp
#    fits[:, 13]=self.info_carriers
#    fits[:,14]=nocl_min
#    fits[:,15]=dipp

    return fits

  def fit_one_model(self, nocl):
    gmm=gaussian_mixture_constr.GaussianMixture(n_components=nocl, n_init=10, covariance_type='spherical') 
    gmm.fit(self.procdata)
    cov=','.join( map(str, np.round(gmm.covariances_, 4)))
    freq=np.round(1-np.max(gmm.weights_))
    wts= ','.join( map(str, np.round(gmm.weights_, 4)))
    bic=gmm.bic(self.procdata)
    lld=gmm.score(self.procdata)
    nn=gmm._n_parameters()
    kk, mm = gmm.get_kk_mm()
    ret=np.array([self.comp_id, self.clus_id, self.clus_dist_id, self.start, self.stop, nocl, bic, mm, kk, cov, wts, freq,  self.cn_med, self.cn_mad], dtype='object')
    #ret=np.array([self.comp_id, self.clus_id, self.clus_dist_id, self.start, self.stop, nocl, bic, mm, kk, cov, wts, self.cn_med, self.cn_mad], dtype='object')
    return ret

  def fit_no_model(self):
    ret=np.array([self.comp_id, self.clus_id, self.clus_dist_id, self.start, self.stop, 1, np.nan, 1, 1, str(np.mean(self.procdata)), '1.0', 0.0, self.cn_med, self.cn_mad], dtype='object')
    #ret=np.array([self.comp_id, self.clus_id, self.clus_dist_id, self.start, self.stop, 1, np.nan, 1, 1, str(np.mean(self.procdata)), '1.0',  self.cn_med, self.cn_mad], dtype='object')
    return ret

