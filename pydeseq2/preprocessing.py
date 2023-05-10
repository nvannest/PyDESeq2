from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd


def deseq2_norm(
    counts: Union[pd.DataFrame, np.ndarray],
    conditions: Union[pd.DataFrame, np.ndarray] = None,
    norm_methods: str = "RLE",
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]:
    """
    Return normalized counts and size_factors.

    Uses the median of ratios method.

    Parameters
    ----------
    counts : pandas.DataFrame or ndarray
        Raw counts. One column per gene, one row per sample.
    conditions : pandas.DataFrame or ndarray
        Raw counts. One column per gene, one row per sample.
    norm_methods : str
        Methods of normalization to apply to the counts matrix.
        Can either be RLE (default) or MRN, which represent
        "Relative Log Expression" or "Median Ratio Normalization".
        Please see "In Papyro Comparison of TMM (edgeR), RLE (DESeq2),
        and MRN Normalization Methods for a Simple Two-Conditions-
        Without-Replicates RNA-Seq Experimental Design"
        (https://www.frontiersin.org/articles/10.3389/fgene.2016.00164/full#B1)
        for more information about MRN.

    Returns
    -------
    deseq2_counts : pandas.DataFrame or ndarray
        DESeq2 normalized counts.
        One column per gene, rows are indexed by sample barcodes.

    size_factors : pandas.DataFrame or ndarray
        DESeq2 normalization factors.
    """
    if norm_methods == "MRN":
        return mrn_normalization(counts, conditions)

    else:
        return rle_normalization(counts)


def rle_normalization(counts):
    """
    This normalization method stands for "Relative Log Expression" and is
    described in "In Papyro Comparison of TMM (edgeR), RLE (DESeq2),
    and MRN Normalization Methods for a Simple Two-Conditions-Without-
    Replicates RNA-Seq Experimental Design" (Maza et al. 2016).

    Parameters
    ----------
    counts : pandas.DataFrame or ndarray
        Raw counts. One column per gene, one row per sample.

    Returns
    -------
    deseq2_counts : pandas.DataFrame or ndarray
        DESeq2 normalized counts.
        One column per gene, rows are indexed by sample barcodes.

    size_factors : pandas.DataFrame or ndarray
        DESeq2 normalization factors.
    """
    # Compute gene-wise mean log counts
    with np.errstate(divide="ignore"):  # ignore division by zero warnings
        log_counts = np.log(counts)
    logmeans = log_counts.mean(0)
    # Filter out genes with -∞ log means
    filtered_genes = ~np.isinf(logmeans)
    # Subtract filtered log means from log counts
    if isinstance(log_counts, pd.DataFrame):
        log_ratios = log_counts.loc[:, filtered_genes] - logmeans[filtered_genes]
    else:
        log_ratios = log_counts[:, filtered_genes] - logmeans[filtered_genes]
    # Compute sample-wise median of log ratios
    log_medians = np.median(log_ratios, axis=1)
    # Return raw counts divided by size factors (exponential of log ratios)
    # and size factors
    size_factors = np.exp(log_medians)
    deseq2_counts = counts / size_factors[:, None]
    return deseq2_counts, size_factors


def mrn_normalization(counts, conditions):
    """
    This normalization method stands for Median Ratio Normalization and is
    described in "Comparison of normalization methods for differential gene
    expression analysis in RNA-Seq experiments" (Maza et al. 2013). In some
    cases it is superior to RLE because it results in fewer false positive
    discoveries of gene expression difference in DE Analysis.

    Most of this code is adapted with permission from Zachary Frair, and was
    a pythonic version of the MRN supplimental script written in R by
    Maza et al. 2013.

    Parameters
    ----------
    counts : pandas.DataFrame or ndarray
        Raw counts. One column per gene, one row per sample.

    Returns
    -------
    deseq2_counts : pandas.DataFrame or ndarray
        DESeq2 normalized counts.
        One column per gene, rows are indexed by sample barcodes.

    size_factors : pandas.DataFrame or ndarray
        DESeq2 normalization factors.
    """

    if not conditions:
        raise ValueError(
            "A Clinical dataframe describing the conditions tested"
            "must be provided to calculate Median Ratio Normalization (MRN)."
        )

    counts = pd.DataFrame(counts)
    totalCounts = counts.sum(axis=0)
    size_factors = totalCounts
    medianRatios = pd.Series([1] * len(conditions), index=size_factors.index)
    if sum(conditions == 1) > 1:
        meanA = np.mean(
            counts.loc[:, conditions == 1].values / totalCounts[conditions == 1], axis=1
        )
    else:
        meanA = counts.loc[:, conditions == 1].values / totalCounts[conditions == 1]
    for i in range(2, max(conditions) + 1):
        if sum(conditions == i) > 1:
            meanB = np.mean(
                counts.loc[:, conditions == i].values / totalCounts[conditions == i],
                axis=1,
            )
        else:
            meanB = counts.loc[:, conditions == i].values / totalCounts[conditions == i]
        meanANot0 = meanA[(meanA > 0) & (meanB > 0)]
        meanBNot0 = meanB[(meanA > 0) & (meanB > 0)]
        ratios = meanBNot0 / meanANot0
        medianRatios[conditions == i] = np.median(ratios)
        size_factors[conditions == i] = (
            medianRatios[conditions == i] * totalCounts[conditions == i]
        )
    medianRatios = medianRatios / np.exp(np.mean(np.log(medianRatios)))
    size_factors = size_factors / np.exp(np.mean(np.log(size_factors)))
    return medianRatios, size_factors
