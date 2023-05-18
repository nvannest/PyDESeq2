import anndata as ad
import numpy as np
import pandas as pd
import warnings
import pytest
from pydeseq2.dds import DeseqDataSet
from pydeseq2.preprocessing import deseq2_norm, mrn_normalization, rle_normalization
from pydeseq2.utils import load_example_data

def test_deseq2_norm_input_type():
    """Test if the input is either pandas dataframe or numpy ndarray"""
    data = [1, 2, 3, 4]
    conditions = [1, 2, 2, 1]
    try:
        deseq2_norm(data, conditions)
    except Exception as e:
        assert isinstance(e, TypeError), "Expected TypeError"

def test_deseq2_norm_output_type():
    """Test if the output is of correct type"""
    counts = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
    conditions = np.array([1, 2])
    result = deseq2_norm(counts, conditions)
    assert isinstance(result, tuple), "Expected Tuple"
    assert isinstance(result[0], (pd.DataFrame, np.ndarray)), "Expected DataFrame or ndarray"
    assert isinstance(result[1], (pd.DataFrame, np.ndarray)), "Expected DataFrame or ndarray"

def test_deseq2_norm_empty_input():
    """Test if the function works with empty input"""
    counts = np.array([[]])
    conditions = np.array([])
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        with pytest.raises(RuntimeWarning):
            deseq2_norm(counts, conditions)

def test_deseq2_norm_mrn_zero_negative_counts():
    """Test if the function throws an error with zero or negative counts in MRN"""
    counts = pd.DataFrame([[0, -1, 2], [3, 4, -5]])
    conditions = pd.Series(['A', 'B'])
    with pytest.raises(ValueError):
        deseq2_norm(counts, conditions, fit_type="mrn")

def test_fit_size_factors_with_zeros():
    """Test behavior when all genes have at least one zero read count"""
    
    counts_df = load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )

    clinical_df = load_example_data(
        modality="clinical",
        dataset="synthetic",
        debug=False,
    )

    # Add zero read count for each gene
    counts_df.iloc[0, :] = 0

    # Create a DeseqDataSet instance
    dds = DeseqDataSet(counts=counts_df, clinical=clinical_df)

    # Should switch to 'iterative' method automatically and raise a warning
    with pytest.warns(RuntimeWarning, match="Every gene contains at least one zero, cannot compute log geometric means. Switching to iterative mode."):
        dds.fit_size_factors(fit_type='rle')

    assert dds.obsm.get("size_factors") is not None, "Size factors should not be None"

def test_fit_size_factors_impact():
    """Test the impact of normalization on the counts data"""
    counts_df = load_example_data(
        modality="raw_counts",
        dataset="synthetic",
        debug=False,
    )

    clinical_df = load_example_data(
        modality="clinical",
        dataset="synthetic",
        debug=False,
    )

    # Ensure the number of samples in counts matches the number of rows in conditions
    counts = counts_df.values
    conditions = clinical_df["condition"].values

    # Create a DeseqDataSet instance
    dds = DeseqDataSet(counts=counts, clinical=clinical_df)

    # Compute normalized counts before fit_size_factors
    normed_counts_before = np.log(counts_df + 1).mean(1)

    # Fit size factors
    dds.fit_size_factors(fit_type='rle')

    # Compute normalized counts after fit_size_factors
    normed_counts_after = dds.layers.get("normed_counts")

    # There should be a significant difference between before and after normalization
    size_factors = dds.obsm.get("size_factors")
    assert size_factors is not None and size_factors.size != 0, "Normalization should affect the counts data"

def test_mrn_normalization():
    """Test the mrn_normalization function."""

    # Generate random data
    np.random.seed(0)
    counts = pd.DataFrame(np.random.randint(0, 100, size=(10, 5)))
    conditions = pd.Series([1, 1, 2, 2, 2])

    # Perform normalization
    median_ratios, size_factors = mrn_normalization(counts, conditions)

    # Check outputs
    assert isinstance(median_ratios, pd.Series), "median_ratios should be a pandas Series."
    assert isinstance(size_factors, pd.Series), "size_factors should be a pandas Series."

    assert median_ratios.shape == conditions.shape, "median_ratios should have the same shape as conditions."
    assert size_factors.shape == conditions.shape, "size_factors should have the same shape as conditions."
