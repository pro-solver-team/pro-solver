from pathlib import Path

data_path = Path(__file__).resolve().parents[1] / 'dataset'

darcy2d_name = 'Darcy_subset'
darcy2d_path =  data_path / f"{darcy2d_name}.hdf5"

reacdiff1d_name = 'ReacDiff_Nu0.5_Rho2.0_subset'
reacdiff1d_path = data_path / f"{reacdiff1d_name}.hdf5"

burgers1d_name = '1D_Burgers_Sols_Nu1.0_subset'
burgers1d_path = data_path / f"{burgers1d_name}.hdf5"
