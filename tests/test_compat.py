import torch
import numpy as np
from dml_benchmark.trainer import train_single_experiment

def run():
    np.random.seed(42)
    x_train = np.random.rand(100, 3).astype(np.float32)
    y_train = np.sum(x_train, axis=1, keepdims=True)
    dydx_train = np.ones((100, 1, 3)).astype(np.float32)
    
    x_test = np.random.rand(20, 3).astype(np.float32)
    y_test = np.sum(x_test, axis=1, keepdims=True)
    dydx_test = np.ones((20, 1, 3)).astype(np.float32)

    res1 = train_single_experiment(
        x_train, y_train, dydx_train,
        x_test, y_test, dydx_test,
        lambda_=1.0, n_epochs=5, batch_size=32, method="dml_fixed", seed=42, pbar=False
    )
    
    dydx_train_mask = np.ones((100, 1, 3)).astype(np.float32)
    dydx_test_mask = np.ones((20, 1, 3)).astype(np.float32)
    
    res3 = train_single_experiment(
        x_train, y_train, dydx_train,
        x_test, y_test, dydx_test,
        lambda_=1.0, n_epochs=5, batch_size=32, method="dml_fixed", seed=42, pbar=False,
        dydx_train_mask=dydx_train_mask, dydx_test_mask=dydx_test_mask
    )
    print("Unmasked res1 test_value_mse:", res1.test_value_mse)
    print("Masked res3 test_value_mse:", res3.test_value_mse)
    assert np.isclose(res1.test_value_mse, res3.test_value_mse)

run()
