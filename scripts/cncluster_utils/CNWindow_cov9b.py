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

  def __init__(self, comb_id, varid, start, stop, cndata, nocl_max, info_carriers, verbose):
    self.data=cndata
    self.varid=varid
    self.nocl_max=nocl_max
    self.combined_id=comb_id
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
    if (dipp>0.9) and (self.info_carriers < 0.01*self.nsamp):
      fits.append(self.fit_no_model())
    else:
      self.add_noise()
      nocl=1
      while nocl < self.nocl_max: 
        res=self.fit_one_model(nocl)
        fits.append(res)
        if self.verbose:
          print('fit_vals\t'+self.combined_id+'\t'+str(self.start)+"\t"+str(self.stop)+"\t"+str(nocl)+"\t"+str(res[6])+"\t"+str(res[7])+"\t"+str(dipp))
        nocl=nocl+1
    fits=np.vstack(fits)
    fits=np.hstack([fits, np.empty([fits.shape[0], 1], dtype='int32'), np.empty([fits.shape[0], 1], dtype='float64')])
    fits[:, 12]=self.info_carriers
    fits[:,13]=dipp
    return fits

  def fit_one_model(self, nocl):
    gmm=gaussian_mixture_constr.GaussianMixture(n_components=nocl, n_init=10, covariance_type='spherical') 
    gmm.fit(self.procdata)
    icl=gmm.icl(self.procdata)
    bic=gmm.bic(self.procdata)
    nn=gmm._n_parameters()
    cov=','.join( map(str, np.round(gmm.covariances_, 4)))
    freq=np.round(1-np.max(gmm.weights_), 3)
    wts= ','.join( map(str, np.round(gmm.weights_, 4)))
    lld=gmm.score(self.procdata)
    kk, mm = gmm.get_kk_mm()
    info='COVAR='+cov+';WTS='+wts+';LLD='+str(round(lld, 3))+';FREQ='+str(freq)
    ret=np.array([self.combined_id, self.varid, self.start, self.stop, nocl, bic, icl, mm, kk, info, self.cn_med, self.cn_mad], dtype='object')  # 11 fields
    return ret

  def fit_no_model(self):
    info='COVAR=.;WTS=.;LLD=.;FREQ=.'
    ret=np.array([self.combined_id, self.varid, self.start, self.stop, 1, np.nan, np.nan, 1, 1, info,  self.cn_med, self.cn_mad], dtype='object')
    return ret
