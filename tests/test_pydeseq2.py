import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import warnings

import tests
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.preprocessing import deseq2_norm, mrn_normalization, rle_normalization
from pydeseq2.utils import load_example_data

# Single-factor tests


def test_deseq(tol=0.02):
    """Test that the outputs of the DESeq2 function match those of the original R
    package, up to a tolerance in relative error.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

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

    r_res = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_res.csv"), index_col=0
    )

    dds = DeseqDataSet(
        counts=counts_df, clinical=clinical_df, design_factors="condition"
    )
    dds.deseq2()

    res = DeseqStats(dds)
    res.summary()
    res_df = res.results_df

    # check that the same p-values are NaN
    assert (res_df.pvalue.isna() == r_res.pvalue.isna()).all()
    assert (res_df.padj.isna() == r_res.padj.isna()).all()

    # Check that the same LFC, p-values and adjusted p-values are found (up to tol)
    assert (
        abs(r_res.log2FoldChange - res_df.log2FoldChange) / abs(r_res.log2FoldChange)
    ).max() < tol
    assert (abs(r_res.pvalue - res_df.pvalue) / r_res.pvalue).max() < tol
    assert (abs(r_res.padj - res_df.padj) / r_res.padj).max() < tol


def test_deseq_no_refit_cooks(tol=0.02):
    """Test that the outputs of the DESeq2 function *without cooks refit*
    match those of the original R package, up to a tolerance in relative error.
    Note: this is just to check that the workflow runs bug-free, as we expect no outliers
    in the synthetic dataset.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

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

    r_res = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_res.csv"), index_col=0
    )

    dds = DeseqDataSet(
        counts=counts_df,
        clinical=clinical_df,
        design_factors="condition",
        refit_cooks=False,
    )
    dds.deseq2()

    res = DeseqStats(dds)
    res.summary()
    res_df = res.results_df

    # check that the same p-values are NaN
    assert (res_df.pvalue.isna() == r_res.pvalue.isna()).all()
    assert (res_df.padj.isna() == r_res.padj.isna()).all()

    # Check that the same LFC, p-values and adjusted p-values are found (up to tol)
    assert (
        abs(r_res.log2FoldChange - res_df.log2FoldChange) / abs(r_res.log2FoldChange)
    ).max() < tol
    assert (abs(r_res.pvalue - res_df.pvalue) / r_res.pvalue).max() < tol
    assert (abs(r_res.padj - res_df.padj) / r_res.padj).max() < tol


def test_lfc_shrinkage(tol=0.02):
    """Test that the outputs of the lfc_shrink function match those of the original
    R package (starting from the same inputs), up to a tolerance in relative error.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())
    r_res = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_res.csv"), index_col=0
    )
    r_shrunk_res = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_lfc_shrink_res.csv"),
        index_col=0,
    )

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

    r_size_factors = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_size_factors.csv"),
        index_col=0,
    ).squeeze()

    r_dispersions = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_dispersions.csv"), index_col=0
    ).squeeze()

    dds = DeseqDataSet(
        counts=counts_df, clinical=clinical_df, design_factors="condition"
    )
    dds.deseq2()
    dds.obsm["size_factors"] = r_size_factors.values
    dds.varm["dispersions"] = r_dispersions.values
    dds.varm["LFC"].iloc[:, 1] = r_res.log2FoldChange.values * np.log(2)

    res = DeseqStats(dds)
    res.summary()
    res.SE = r_res.lfcSE * np.log(2)
    res.lfc_shrink(coeff="condition_B_vs_A")
    shrunk_res = res.results_df

    # Check that the same LFC are found (up to tol)
    assert (
        abs(r_shrunk_res.log2FoldChange - shrunk_res.log2FoldChange)
        / abs(r_shrunk_res.log2FoldChange)
    ).max() < tol


def test_iterative_size_factors(tol=0.02):
    """Test that the outputs of the iterative size factor method match those of the
    original R package (starting from the same inputs), up to a tolerance in relative
    error.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())
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

    r_size_factors = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_iterative_size_factors.csv"),
        index_col=0,
    ).squeeze()

    dds = DeseqDataSet(
        counts=counts_df, clinical=clinical_df, design_factors="condition"
    )
    dds._fit_iterate_size_factors()

    # Check that the same LFC are found (up to tol)
    assert (
        abs(r_size_factors - dds.obsm["size_factors"]) / abs(r_size_factors)
    ).max() < tol


# Multi-factor tests
def test_multifactor_deseq(tol=0.02):
    """Test that the outputs of the DESeq2 function match those of the original R
    package, up to a tolerance in relative error.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

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

    r_res = pd.read_csv(
        os.path.join(test_path, "data/multi_factor/r_test_res.csv"), index_col=0
    )

    dds = DeseqDataSet(
        counts=counts_df, clinical=clinical_df, design_factors=["group", "condition"]
    )
    dds.deseq2()

    res = DeseqStats(dds)
    res.summary()
    res_df = res.results_df

    # check that the same p-values are NaN
    assert (res_df.pvalue.isna() == r_res.pvalue.isna()).all()
    assert (res_df.padj.isna() == r_res.padj.isna()).all()

    # Check that the same LFC, p-values and adjusted p-values are found (up to tol)
    assert (
        abs(r_res.log2FoldChange - res_df.log2FoldChange) / abs(r_res.log2FoldChange)
    ).max() < tol
    assert (abs(r_res.pvalue - res_df.pvalue) / r_res.pvalue).max() < tol
    assert (abs(r_res.padj - res_df.padj) / r_res.padj).max() < tol


def test_multifactor_lfc_shrinkage(tol=0.02):
    """Test that the outputs of the lfc_shrink function match those of the original
    R package (starting from the same inputs), up to a tolerance in relative error.
    """

    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())
    r_res = pd.read_csv(
        os.path.join(test_path, "data/multi_factor/r_test_res.csv"), index_col=0
    )
    r_shrunk_res = pd.read_csv(
        os.path.join(test_path, "data/multi_factor/r_test_lfc_shrink_res.csv"),
        index_col=0,
    )

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

    r_size_factors = pd.read_csv(
        os.path.join(test_path, "data/multi_factor/r_test_size_factors.csv"),
        index_col=0,
    ).squeeze()

    r_dispersions = pd.read_csv(
        os.path.join(test_path, "data/multi_factor/r_test_dispersions.csv"), index_col=0
    ).squeeze()

    dds = DeseqDataSet(
        counts=counts_df, clinical=clinical_df, design_factors=["group", "condition"]
    )
    dds.deseq2()
    dds.obsm["size_factors"] = r_size_factors.values
    dds.varm["dispersions"] = r_dispersions.values
    dds.varm["LFC"].iloc[:, 1] = r_res.log2FoldChange.values * np.log(2)

    res = DeseqStats(dds)
    res.summary()
    res.SE = r_res.lfcSE * np.log(2)
    res.lfc_shrink(coeff="condition_B_vs_A")
    shrunk_res = res.results_df

    # Check that the same LFC found (up to tol)
    assert (
        abs(r_shrunk_res.log2FoldChange - shrunk_res.log2FoldChange)
        / abs(r_shrunk_res.log2FoldChange)
    ).max() < tol


def test_contrast():
    """
    Check that the contrasts ['condition', 'B', 'A'] and ['condition', 'A', 'B'] give
    coherent results (change of sign in LFCs and Wald stats, same (adjusted p-values).
    """

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

    dds = DeseqDataSet(
        counts=counts_df, clinical=clinical_df, design_factors=["group", "condition"]
    )
    dds.deseq2()

    res_B_vs_A = DeseqStats(dds, contrast=["condition", "B", "A"])
    res_A_vs_B = DeseqStats(dds, contrast=["condition", "A", "B"])
    res_B_vs_A.summary()
    res_A_vs_B.summary()

    # Check that all values correspond, up to signs
    for col in res_B_vs_A.results_df.columns:
        np.testing.assert_almost_equal(
            res_B_vs_A.results_df[col].abs().values,
            res_A_vs_B.results_df[col].abs().values,
            decimal=8,
        )

    # Check that the sign of LFCs and stats are inverted
    np.testing.assert_almost_equal(
        res_B_vs_A.results_df.log2FoldChange.values,
        -res_A_vs_B.results_df.log2FoldChange.values,
        decimal=8,
    )
    np.testing.assert_almost_equal(
        res_B_vs_A.results_df.stat.values, -res_A_vs_B.results_df.stat.values, decimal=8
    )


def test_anndata_init(tol=0.02):
    """
    Test initializing dds with an AnnData object that already has filled in fields,
    including with the same names as those used by pydeseq2.
    """
    np.random.seed(42)

    # Load data
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

    # Make an anndata object
    adata = ad.AnnData(X=counts_df.astype(int), obs=clinical_df)

    # Put some dummy data in unused fields
    adata.obsm["dummy_metadata"] = np.random.choice(2, adata.n_obs)
    adata.varm["dummy_param"] = np.random.randn(adata.n_vars)

    # Put values in the dispersions field
    adata.varm["dispersions"] = np.random.randn(adata.n_vars) ** 2

    # Load R data
    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    r_res = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_test_res.csv"), index_col=0
    )

    # Initialize DeseqDataSet from anndata and test results
    dds = DeseqDataSet(adata=adata, design_factors="condition")
    dds.deseq2()

    res = DeseqStats(dds)
    res.summary()
    res_df = res.results_df

    # check that the same p-values are NaN
    assert (res_df.pvalue.isna() == r_res.pvalue.isna()).all()
    assert (res_df.padj.isna() == r_res.padj.isna()).all()

    # Check that the same LFC, p-values and adjusted p-values are found (up to tol)
    assert (
        abs(r_res.log2FoldChange - res_df.log2FoldChange) / abs(r_res.log2FoldChange)
    ).max() < tol
    assert (abs(r_res.pvalue - res_df.pvalue) / r_res.pvalue).max() < tol
    assert (abs(r_res.padj - res_df.padj) / r_res.padj).max() < tol


def test_vst(tol=0.02):
    """
    Test the output of VST compared with DESeq2.
    """

    # Load data
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

    # Load R data
    test_path = str(Path(os.path.realpath(tests.__file__)).parent.resolve())

    r_vst = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_vst.csv"), index_col=0
    ).T

    r_vst_with_design = pd.read_csv(
        os.path.join(test_path, "data/single_factor/r_vst_with_design.csv"), index_col=0
    ).T

    # Test blind design
    dds = DeseqDataSet(
        counts=counts_df, clinical=clinical_df, design_factors=["condition"]
    )
    dds.vst(use_design=False)
    assert (np.abs(r_vst - dds.layers["vst_counts"]) / r_vst).max().max() < tol

    # Test full design
    dds = DeseqDataSet(
        counts=counts_df, clinical=clinical_df, design_factors=["condition"]
    )
    dds.vst(use_design=True)
    assert (
        np.abs(r_vst_with_design - dds.layers["vst_counts"]) / r_vst_with_design
    ).max().max() < tol


def test_ref_level():
    """Test that DeseqDataSet columns are created according to the passed reference
    level, if any.
    """
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

    dds = DeseqDataSet(
        counts=counts_df,
        clinical=clinical_df,
        design_factors=["group", "condition"],
        ref_level=["group", "Y"],
    )

    # Check that the column exists
    assert "group_X_vs_Y" in dds.obsm["design_matrix"].columns
    # Check that its content is correct
    assert (
        dds.obsm["design_matrix"]["group_X_vs_Y"]
        == clinical_df["group"].apply(
            lambda x: 1 if x == "X" else 0 if x == "Y" else np.NaN
        )
    ).all()


def test_deseq2_norm():
    """Test that deseq2_norm() called on a pandas dataframe outputs the same results as
    DeseqDataSet.fit_size_factors()
    """
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

    # Fit size factors from DeseqDataSet
    dds = DeseqDataSet(counts=counts_df, clinical=clinical_df)
    dds.fit_size_factors()
    s1 = dds.obsm["size_factors"]

    # Fit size factors from counts directly
    s2 = deseq2_norm(counts_df)[1]

    np.testing.assert_almost_equal(
        s1,
        s2,
        decimal=8,
    )

# Normalization/preprocessing tests

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
        dds.fit_size_factors()

    assert dds.obsm.get("size_factors") is not None, "Size factors should not be None"

def test_fit_size_factors_dimpact():
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
    dds.fit_size_factors()

    # Compute normalized counts after fit_size_factors
    normed_counts_after = dds.layers.get("normed_counts")

    assert np.any(normed_counts_before != normed_counts_after), "Normalization should change the counts data"

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
