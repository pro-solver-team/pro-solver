import h5py
from pro_solver.benchmark.data_modules.data_vars import (
    darcy2d_path,
    reacdiff1d_path,
    burgers1d_path,
    advection1d_path,
    compressible_ns1d_path,
    incomp_ns2d_path,
    shallow_water2d_path,
    diffsorp1d_path,
    reacdiff2d_path
)

def equation_to_numpy(equation_name: str):
    if equation_name == 'darcy2d':
        file_path = darcy2d_path
        with h5py.File(file_path, 'r') as f:
            features = f['nu'][:]
            target = f['tensor'][:]
            x = f['x-coordinate'][:]
            y = f['y-coordinate'][:]
        return features, x, y, target

    elif equation_name == 'rec_diff':
        file_path = reacdiff1d_path
        with h5py.File(file_path, 'r') as f:
            t = f['t-coordinate'][:-1]
            x = f['x-coordinate'][:]
            target = f['tensor'][:]
            u_0 = target[:, 0]
        return t, x, u_0, target

    elif equation_name == 'burgers1d':
        file_path = burgers1d_path
        with h5py.File(file_path, 'r') as f:
            t = f['t-coordinate'][:]
            x = f['x-coordinate'][:]
            target = f['tensor'][:]
            u_0 = target[:, 0]
        return t, x, u_0, target

    elif equation_name == 'advection1d':
        file_path = advection1d_path
        with h5py.File(file_path, 'r') as f:
            t = f['t-coordinate'][:]
            x = f['x-coordinate'][:]
            target = f['tensor'][:]
            u_0 = target[:, 0]
        return t, x, u_0, target

    elif equation_name == 'compressible_ns1d':
        file_path = compressible_ns1d_path
        with h5py.File(file_path, 'r') as f:
            t = f['t-coordinate'][:]
            x = f['x-coordinate'][:]
            target = f['tensor'][:]
            u_0 = target[:, 0]
        return t, x, u_0, target

    elif equation_name == 'incomp_ns2d':
        file_path = incomp_ns2d_path
        with h5py.File(file_path, 'r') as f:
            t = f['t-coordinate'][:]
            x = f['x-coordinate'][:]
            y = f['y-coordinate'][:]
            target = f['tensor'][:]
            u_0 = target[:, 0]
        return t, x, y, u_0, target

    elif equation_name == 'shallow_water2d':
        file_path = shallow_water2d_path
        with h5py.File(file_path, 'r') as f:
            t = f['t-coordinate'][:]
            x = f['x-coordinate'][:]
            y = f['y-coordinate'][:]
            target = f['tensor'][:]
            u_0 = target[:, 0]
        return t, x, y, u_0, target

    elif equation_name == 'diff_sorp1d':
        file_path = diffsorp1d_path
        with h5py.File(file_path, 'r') as f:
            t = f['t-coordinate'][:]
            x = f['x-coordinate'][:]
            target = f['tensor'][:]
            u_0 = target[:, 0]
        return t, x, u_0, target

    elif equation_name == 'rec_diff2d':
        file_path = reacdiff2d_path
        with h5py.File(file_path, 'r') as f:
            t = f['t-coordinate'][:]
            x = f['x-coordinate'][:]
            y = f['y-coordinate'][:]
            target = f['tensor'][:]
            u_0 = target[:, 0]
        return t, x, y, u_0, target

    else:
        raise ValueError("no equation dodik")


if __name__ == "__main__":
    t,x,u0,tar = equation_to_numpy('rec_diff')
