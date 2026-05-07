import h5py
import numpy as np

with h5py.File("./data/pdebench/2D_DarcyFlow_beta1.0_Train.hdf5", "r") as f:
    u_all = np.asarray(f["tensor"][0, 0, :, :], dtype=np.float32)
    a_all = np.asarray(f["nu"][0, :, :], dtype=np.float32)
    x_coords = np.asarray(f["x-coordinate"][:], dtype=np.float32)
    y_coords = np.asarray(f["y-coordinate"][:], dtype=np.float32)

dx = float(x_coords[1] - x_coords[0])
dy = float(y_coords[1] - y_coords[0])

dudx = np.zeros_like(u_all)
dudx[1:-1, :] = (u_all[2:, :] - u_all[:-2, :]) / (2 * dx)
dudx[0, :] = (u_all[1, :] - u_all[0, :]) / dx
dudx[-1, :] = (u_all[-1, :] - u_all[-2, :]) / dx

dudy = np.zeros_like(u_all)
dudy[:, 1:-1] = (u_all[:, 2:] - u_all[:, :-2]) / (2 * dy)
dudy[:, 0] = (u_all[:, 1] - u_all[:, 0]) / dy
dudy[:, -1] = (u_all[:, -1] - u_all[:, -2]) / dy

print(f"X coords: {x_coords.shape}, dx: {dx}")
print(f"u shape: {u_all.shape}, dtype: {u_all.dtype}, min: {u_all.min()}, max: {u_all.max()}, mean: {u_all.mean()}")
print(f"a shape: {a_all.shape}, dtype: {a_all.dtype}, min: {a_all.min()}, max: {a_all.max()}, mean: {a_all.mean()}")
print(f"dudx stats - max: {dudx.max():.4f}, min: {dudx.min():.4f}, mean: {dudx.mean():.4f}, std: {dudx.std():.4f}")
print(f"dudy stats - max: {dudy.max():.4f}, min: {dudy.min():.4f}, mean: {dudy.mean():.4f}, std: {dudy.std():.4f}")

