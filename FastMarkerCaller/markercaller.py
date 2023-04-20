from typing import Union
from itertools import combinations
import numpy as np
import pandas as pd
import warnings
from .fastmarker import FastMarkerCaller
from anndata import AnnData
from typing import Union, Iterable
import multiprocessing
import functools

class MarkerCaller(FastMarkerCaller):
    """
    A class for calling markers in single-cell RNA-seq data.

    Parameters:
        adata (AnnData): An annotated data matrix with rows for cells and columns for genes.
        groupby (str): The key for cell annotations used for grouping cells into different categories.

    Attributes:
        AUROC_DOWNSAMPLE_WARNING (int): The size threshold that triggers a warning message for estimating AUROC.

    Methods:
        call_one_vs_rest: Calls markers between each group and the rest of the data.
        call_pairwise: Calls pairwise markers between each pair of groups.
    """

    AUROC_DOWNSAMPLE_WARNING: int = 500

    def __init__(self, adata: AnnData, groupby: str) -> None:
        """Initializes MarkerCaller with anndata object and groupby parameter."""
        super().__init__(adata, groupby)

    def _check_auroc_downsample(self, **kwargs: dict) -> None:
        """Checks whether to downsample when computing AUROC."""
        if 'auroc_downsample' not in kwargs and len(self.adata) > self.AUROC_DOWNSAMPLE_WARNING:
            if kwargs.get('auroc', True):
                warnings.warn('Computing AUROC for markers is time consuming. '
                              'Please consider to turn off auroc computation by setting auroc=False '
                              'or to use downsampling auroc approximation by setting auroc_downsample=N'
                              ' to accelerate the computation.')

    def _clean_args(self, **kwargs: dict) -> dict:
        """Removes groups1 and groups2 from dictionary of keyword arguments."""
        if 'groups1' in kwargs:
            kwargs.pop('groups1')
        if 'groups2' in kwargs:
            kwargs.pop('groups2')
        return kwargs

    def call_markers(self, groups1:Union[Iterable,str], groups2:Union[Iterable,str]=None, 
            groups1_name:str=None, groups2_name:str=None, **kwargs: dict) -> pd.DataFrame:
        """Calls markers between two groups.
            Parameters
                groups1 (iterable or str): One or more groups for the first category.
                groups2 (iterable or str or None): One or more groups for second group. If not provided
                    all other groups will be considered for comparison.
                groups1_name: bool, optional (default=False)
                groups1_name: bool, optional (default=False)
                kwargs: dict, optional
                    See call_marker in FastMarkerCaller for kwargs details
            Returns
                summary: pandas.DataFrame
                    A summary table of the pairwise markers"""

        #TODO
        if not isinstance(groups1, str) and groups1_name is None:
            raise ValueError()
        if not isinstance(groups2, str) and groups2_name is None:
            raise ValueError()
        if isinstance(groups1, str) and groups1_name is None:
            groups1_name = groups1
        if isinstance(groups2, str) and groups2_name is None:
            groups2_name = groups2

        markers = super().call_markers(groups1, groups2, **kwargs)
        markers['fg'] = groups1_name
        markers['bg'] = groups2_name
        return markers

    def call_one_vs_rest(self, n_jobs=1, **kwargs: dict) -> pd.DataFrame:
        """Calls markers between each group and the rest of the data.
            Parameters
                kwargs: dict, optional
                    See call_marker in FastMarkerCaller for kwargs details
            Returns
                summary: pandas.DataFrame
                    A summary table of the one-vs-rest markers."""
        self._clean_args(**kwargs)
        self._check_auroc_downsample(**kwargs)

        worker = functools.partial(self.call_markers, groups2_name='_REST_', **kwargs)
        if n_jobs>1:
            with multiprocessing.Pool(n_jobs) as pool:
                summary = pool.map(worker, self.group_names)
        else:
            summary = map(worker, self.group_names)
        #summary = []
        #for grp in self.group_names:
        #    #tmp = self.call_markers(groups1=grp, groups2=None, **kwargs)
        #    #tmp['fg'] = grp
        #    #tmp['bg'] = '_REST_'
        #    tmp = self._call_markers(groups1=grp, groups2=None, groups1_name=grp, groups2_name='_REST_', **kwargs)
        #    summary.append(tmp)
        summary = pd.concat(summary)
        return summary

    def _get_group_pairs(self, force_all_pairs: bool, max_pairs: int, **kwargs: dict) -> np.ndarray:
        """Gets all possible group pairs or a subset of random group pairs depending on parameters."""
        pairs = np.array(list(combinations(self.group_names, 2)))
        if len(pairs) > max_pairs:
            if not force_all_pairs:
                sel = sorted(np.random.choice(len(pairs), max_pairs))
                pairs = pairs[sel]
                warnings.warn(f'The cluster pair number is reduced from {len(pairs)} to {max_pairs} by donwsampling.')
            else:
                warnings.warn(f'The number ({len(pairs)}) of cluster pairs is too large, '
                              'which may take very long time to compute. '
                              'Please consider set force_all_pairs=False and set max_pairs<=500 '
                              'to compute only a subset of cluster pairs.')
        return pairs

    def call_pairwise(self, force_all_pairs: bool = False, max_pairs: int = 500, two_sided: bool = False, n_jobs=1, **kwargs: dict) -> pd.DataFrame:
        """Calls markers between each pair of groups.
            Parameters
                force_all_pairs: bool, optional (default=False)
                    If True, then all pairs will be called regardless of total pair number. 
                    If False, then only max_pairs randomly selected pairs will be called.
                max_pairs: int, optional (default=500)
                    The maximum number of pairwise comparisons to make if force_all_pairs is False.
                    Pairs will be randomly selected from all possible cluster pairs.
                two_sided: bool, optional (default=False)
                    If True, then both upregulated and downregulated markers are returned.
                      If False, then only upregulated markers are returned.
                kwargs: dict, optional
                    See call_marker in FastMarkerCaller for kwargs details
            Returns
                summary: pandas.DataFrame
                    A summary table of the pairwise markers"""
        self._clean_args(**kwargs)
        self._check_auroc_downsample(**kwargs)
        kwargs.update({'two_sided':True})
        pairs = self._get_group_pairs(force_all_pairs, max_pairs)

        #def worker(abc):
        #    (groups1,groups2),kwargs = abc
        #    return self._call_markers(groups1, groups2,**kwargs)

        worker = functools.partial(self.call_markers, **kwargs)
        if n_jobs > 1:
            with multiprocessing.Pool(n_jobs) as pool:
                summary = pool.starmap(worker, pairs)
        else:
            g1s, g2s = zip(*pairs)
            summary = list(map(worker, g1s, g2s))

        for i in range(len(pairs)):
            tmp12 = summary[i]
        
            tmp21 = tmp12.copy()
            tmp21['diff'] = -tmp21['diff']
            tmp21['fc'] = 1/tmp21['fc']
            tmp21['fg'] = tmp12['bg']
            tmp21['bg'] = tmp12['fg']
            if not two_sided:
                tmp12 = tmp12[tmp12['diff']>0]
                tmp21 = tmp21[tmp21['diff']>0]
            summary[i] = tmp12
            summary.append(tmp21)
           
        #for grp1,grp2 in pairs:
        #    tmp12 = self.call_markers(groups1=grp1, groups2=grp2, **kwargs)
        #    tmp12['fg'] = grp1
        #    tmp12['bg'] = grp2
        #
        #    tmp21 = tmp12.copy()
        #    tmp21['diff'] = -tmp21['diff']
        #    tmp21['fc'] = 1/tmp21['fc']
        #    tmp21['fg'] = grp2
        #    tmp21['bg'] = grp1
        #    if not two_sided:
        #        tmp12 = tmp12[tmp12['diff']>0]
        #        tmp21 = tmp21[tmp21['diff']>0]
        #    summary.append(tmp12)
        #    summary.append(tmp21)
        summary = pd.concat(summary)
        return summary
