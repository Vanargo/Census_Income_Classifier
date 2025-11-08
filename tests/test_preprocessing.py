# Copyright (c) 2025 Vanargo
# Licensed under the MIT License. See LICENSE in the project root.

import numpy as np
import pytest
import scipy.sparse as sp
from joblib import load


def _get_preproc_from_model(model):
    # sklearn Pipeline #
    if hasattr(model, "named_steps"):
        return model.named_steps.get("preproc") or model.named_steps.get("preprpocessor")
    # container-like object with attributes #
    for attr in ("preproc", "preprocessor"):
        if hasattr(model, attr):
            return getattr(model, attr)
    return None


@pytest.mark.smoke
def test_preprocessor_smoke(model_path, small_raw_df):
    model = load(model_path)
    preproc = _get_preproc_from_model(model)
    assert preproc is not None, "Preprocessor is required in the saved pipeline for this test."

    X = preproc.transform(small_raw_df)

    # basic invariants #
    assert X is not None
    assert getattr(X, "shape", None) is not None
    assert X.shape[0] == len(small_raw_df)
    assert X.shape[1] > 0

    # no NaNs #
    if sp.issparse(X):
        assert not np.isnan(X.data).any()
    else:
        arr = np.asarray(X)
        assert not np.isnan(arr).any()

    # determinism #
    X2 = preproc.transform(small_raw_df)
    if sp.issparse(X) and sp.issparse(X2):
        assert (X != X2).nnz == 0
    else:
        np.testing.assert_allclose(np.asarray(X), np.asarray(X2), rtol=0, atol=0)
