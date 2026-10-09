import numpy as np
import pytest
from scipy import sparse

from feature_engine.transformation import LogTransformer
from tests.estimator_checks.sklearn_check_wrapper import wrap_for_check_estimator

_sparse_containers = [sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix]
if hasattr(sparse, "csr_array"):
    _sparse_containers.append(sparse.csr_array)


@pytest.mark.parametrize("sparse_container", _sparse_containers)
@pytest.mark.parametrize("method", ["fit", "transform", "inverse_transform"])
def test_wrapper_rejects_sparse_input(sparse_container, method):
    # Match check_X's sparse-input contract before numpy conversion can turn
    # sparse data into a scalar object. Its text varies with scipy's repr.
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    estimator = wrap_for_check_estimator(LogTransformer()).fit(X)

    with pytest.raises(TypeError, match="sparse"):
        getattr(estimator, method)(sparse_container(X))
