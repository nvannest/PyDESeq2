from typing import Literal
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd


def deseq2_norm(
    counts: Union[pd.DataFrame, np.ndarray],
    conditions: Union[pd.DataFrame, np.ndarray] = None,
    fit_type: Literal["rle", "mrn"] = "rle",
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
    if fit_type == "mrn":
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
    Normalize counts data using the Median Ratio Normalization method.

    Parameters
    ----------
    counts : pandas.DataFrame or numpy.ndarray
        Raw counts. One column per sample, one row per gene.
    conditions : pandas.Series or numpy.ndarray
        Conditions for each sample.

    Returns
    -------
    median_ratios : pandas.Series or numpy.ndarray
        Median ratios for each condition.
    size_factors : pandas.Series or numpy.ndarray
        Normalization factors for each sample.
    """
    pandas_input = isinstance(counts, pd.DataFrame)

    if isinstance(counts, np.ndarray):
        counts = pd.DataFrame(counts)
    if isinstance(conditions, np.ndarray):
        conditions = pd.Series(conditions)
        
    if counts.empty or conditions.empty:
        raise ValueError("Counts and conditions must not be empty.")

    for i in conditions.unique():
        meanA = _calculate_mean(counts, conditions, 1)
        meanB = _calculate_mean(counts, conditions, i)
        meanANot0 = meanA[(meanA > 0) & (meanB > 0)]
        meanBNot0 = meanB[(meanA > 0) & (meanB > 0)]
        ratios = meanBNot0 / meanANot0

        median_ratios[conditions == i] = np.median(ratios)
        size_factors[conditions == i] = median_ratios[conditions == i] * totalCounts[conditions == i]

    # normalize ratios and size_factors
    median_ratios = median_ratios / np.exp(np.mean(np.log(median_ratios)))
    size_factors = size_factors / np.exp(np.mean(np.log(size_factors)))

    if not pandas_input:
        median_ratios = median_ratios.values
        size_factors = size_factors.values

    return median_ratios, size_factors



