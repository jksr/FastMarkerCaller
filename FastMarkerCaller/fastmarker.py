import pandas as pd
import numpy as np
import scipy.stats as ss
from scipy.sparse import issparse
import anndata
from statsmodels.stats.multitest import fdrcorrection
from numba import jit
from .group_obs_mean import group_obs_mean
# from collections.abc import Iterable
from typing import Union, Iterable

def fast_auc(y_true:Iterable[bool], y_prob:Iterable[float], downsample:int=None) -> np.array:
    """
    Compute the area under the curve (AUC) score for multiple features.

    Args:
        y_true: 1d binary labels of size n.
        y_prob: Predicted probability matrix of size m x n.
        downsample: downsample to this size to accelerate auroc computation
    Returns:
        AUC scores for each predicted probability set. In total, n AUC scores will be calculated.
    """
    if issparse(y_prob):
        y_prob = y_prob.todense()

    nvar,nobs = y_prob.shape
    y_true = np.tile(y_true, (nvar,1))
    y_prob = np.asarray(y_prob)
    if downsample is not None and nobs>downsample:
        sel = np.random.choice(nobs, downsample)
        y_true = y_true[:,sel]
        y_prob = y_prob[:,sel]
    

    #https://gist.github.com/r0mainK/9ecce4b2a9352ca3d070a19ce43d7f1a
    y_true = y_true[np.arange(nvar).reshape(-1,1),
                    np.argsort(y_prob, axis=1)].astype(float)
    false_count = np.cumsum(1 - y_true, axis=1)
    aucs = (y_true * false_count).sum(1)
    aucs = aucs / (false_count[:, -1] * (y_true.shape[1] - false_count[:,-1]))
    return aucs


def welch_t(mean1:float, mean2:float, sesq1:float, sesq2:float, n1:int, n2:int, two_sided:bool) -> float:
    """
    Perform Welch's t-test to compare means from two independent group_names with unequal variances.

    Args:
        mean1: Mean of group 1.
        mean2: Mean of group 2.
        sesq1: Squared standard error of group 1.
        sesq2: Squared standard error of group 2.
        n1: Number of samples in group 1.
        n2: Number of samples in group 2.
        two_sided: Two sided test or not

    Returns:
        The t-statistic and the p-value for the two-sided test.
    """
    t = (mean1-mean2)/((sesq1+sesq2)**0.5)
    v = (sesq1+sesq2)**2/( sesq1**2/(n1-1) + sesq2**2/(n2-1) )
    
    ## use sf instead of cdf, which is not accurate enough
    # https://stackoverflow.com/questions/6298105/precision-of-cdf-in-scipy-stats
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    
    #still has problem
    
#     p = (1 - ss.t.cdf(abs(t), v)) * 2
    p = ss.t.sf(abs(t), v)
    if two_sided:
        p *= 2
    return t, p
    
def nafdrcorrection(p:Iterable[float], alpha:float=0.05, method:str='indep', is_sorted:bool=False) -> tuple:
    """
    Apply the false discovery rate correction on p-values.

    Args:
        p: p-values to be corrected.
        alpha: Significance level.
        method: Method of correction. Default is 'indep'.
        is_sorted: Whether input array is sorted.

    Returns:
        Tuple containing a bool array for whether p-values are significant and another array for their adjusted values.
    """
    ids = ~np.isnan(p)
    adjp = np.array([np.nan]*len(p))
    sig = np.array([False]*len(p))
    fdr_rlt = fdrcorrection(p[ids],alpha,method,is_sorted)
    sig[ids] = fdr_rlt[0]
    adjp[ids] = fdr_rlt[1]
    return sig, adjp
      
class FastMarkerCaller:
    """
    A class FastMarkerCaller that performs independent two-sample t-test and returns significant markers 
    among user-defined groups
    
    Args:
        adata: AnnData object, contains gene expression data, with genes in row and cells in columns.
        groupby: str, column key name for a categorical grouping information of cells.

    Attributes:
        sizes: The number of cells in each group. Size of each group provided by groupby variable.
        means: Mean expression level of each gene across cell of the same group list.
        stds: Standard deviations of gene expression values within a given group.
        sqstderrs: Mean squared standard error for each gene.
        feat_names: Names of the features (e.g., genes)
        group_names: Names of the groups obtained from `groupby`
        adata: AnnData object used in instantiating the object.
        groupby: name of the grouping variable used in instantiating the object.

    Methods:
        call_markers: Perform Welch's independent two-sample t-test on different groups and returns 
            significant upregulated/deactivated gene(s)

    Example:
    #### Initialize FastMarkerCaller object on example dataset ####
    fastcaller = FastMarkerCaller(adata, 'batch')
    #### Using the object to perform marker test ####
    res = fastcaller.call_markers(['batch1'], ['batch2'], fdr=0.05)
    string_res = res.to_string()
    print(string_res)
        
    """
    def __init__(self, adata:anndata.AnnData, groupby:str, ) -> None:
        self.sizes = adata.obs[groupby].value_counts(sort=False)
        self.means = group_obs_mean(adata, groupby)[self.sizes.index]
        self.stds = group_obs_mean(adata, groupby, np.std)[self.sizes.index]
        self.sqstderrs = self.stds**2/self.sizes

        self.feat_names = adata.var_names
        self.group_names = self.sizes.index
        
        self.adata = adata # save for auroc 
        self.groupby = groupby
        
    def _grp_info(self, groups:Iterable[str]) -> tuple:
        """Extract group sizes, group means, and group squared standard errors"""

        ind_sizes = self.sizes.loc[groups]
        ind_means = self.means[groups]
        ind_sqstderrs = self.sqstderrs[groups]
        mg_mean = (ind_means*ind_sizes).sum(1)/ind_sizes.sum()    
        mg_sqstderr = (ind_sqstderrs*ind_sizes*(ind_sizes-1)).sum(1)/sum(ind_sizes)/sum(ind_sizes)
        mg_size = ind_sizes.sum()
        return mg_size, mg_mean, mg_sqstderr
        
    
    def call_markers(self, groups1:Union[Iterable,str], groups2:Union[Iterable,str]=None, *, 
                     fdr:float=0.05, topn:int=None, two_sided:bool=False, 
                     auroc:bool=True, auroc_cutoff:float=0.60, 
                     auroc_downsample:int=None) -> pd.DataFrame:
        
        """
        Parameters:
            groups1 (iterable or str): One or more groups for the first category.
            groups2 (iterable or str or None): One or more groups for second group. If not provided
                all other groups will be considered for comparison.
            fdr (float): False discovery rate cut-off. Default is 0.05.
            topn (int): Number of top significant genes returned. Default is None (all significant).
            two_sided (boolean): Whether to look at two-sided t-test or one-sided (over-expressed).
                Default is False. 
            auroc (boolean): Whether to calculate AUROC scores for significant genes using 
                the entire dataset. Default is True.
            auroc_cutoff (float): Cutoff for the minimum value of AUROC score. Default is 0.60.
            auroc_downsample (int): Only use subsamples for AUROC calculation, 
                since it is the slowest step. Use all samples if not specified. Default is None.

        Returns:
            pandas.DataFrame: A DataFrame containing summary of markers determined.
            Rows may contain the following columns: p_val, adj_pval, diff, fc, significant and auroc

        """
        if isinstance(groups1, str) or not isinstance(groups1, Iterable):
            groups1 = [groups1]
        if groups2 is None:
            groups2 = self.group_names[~self.group_names.isin(groups1)].tolist()
        if isinstance(groups2, str):
            groups2 = [groups2]
        grps1_size, grps1_mean,grps2_sqstderr = self._grp_info(groups1)
        grps2_size, grps2_mean,grps2_sqstderr = self._grp_info(groups2)
        
        t, p = welch_t(grps1_mean, grps2_mean, grps2_sqstderr, grps2_sqstderr, 
                       grps1_size, grps2_size, two_sided)
        sig, adjp = nafdrcorrection(p, fdr)
        
        delta = grps1_mean-grps2_mean
        fc = grps1_mean/grps2_mean
        
        summary = pd.DataFrame({'welch-t':pd.Series(t,index=self.feat_names, dtype=float),
                                'p-val':pd.Series(p,index=self.feat_names, dtype=float),
                                'adj_p-val':pd.Series(adjp,index=self.feat_names, dtype=float),
                                'diff':pd.Series(delta,index=self.feat_names, dtype=float),
                                'fc':pd.Series(fc, index=self.feat_names, dtype=float),
                                'significant':pd.Series(sig,index=self.feat_names, dtype=bool),
                                })
        # add dtypes to prevent unexpected issues such as https://github.com/pandas-dev/pandas/issues/46292
        
        summary = summary.sort_values('welch-t', ascending=False)
        #summary = summary.sort_values('adj_p-val')
        summary = summary[summary['adj_p-val']<=fdr]
        if not two_sided:
            summary = summary[summary['diff']>0]
        if topn is not None:
            summary = summary.head(topn)
            
        if auroc:
            adata = self.adata[self.adata.obs[self.groupby].isin(groups1+groups2)]
            isgroups1 = adata.obs[self.groupby].isin(groups1)
            summary['AUROC'] = fast_auc(isgroups1, adata[:, summary.index].X.T,
                                        auroc_downsample)
            summary = summary[summary['AUROC']>auroc_cutoff]
            
        return summary
        
