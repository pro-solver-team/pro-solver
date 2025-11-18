from pathlib import Path

data_path = Path(__file__).resolve().parents[1] / 'dataset'

darcy2d_name = 'Darcy_subset'
darcy2d_path =  data_path / f"{darcy2d_name}.hdf5"

reacdiff1d_name = 'ReacDiff_Nu0.5_Rho2.0_subset'
reacdiff1d_path = data_path / f"{reacdiff1d_name}.hdf5"

burgers1d_name = '1D_Burgers_Sols_Nu1.0_subset'
burgers1d_path = data_path / f"{burgers1d_name}.hdf5"

advection1d_name = '1D_Advection_Sols_C1.0_subset'
advection1d_path = data_path / f"{advection1d_name}.hdf5"

compressible_ns1d_name = '1D_CompNS_Sols_subset'
compressible_ns1d_path = data_path / f"{compressible_ns1d_name}.hdf5"

incomp_ns2d_name = 'IncompNS2D_subset'
incomp_ns2d_path = data_path / f"{incomp_ns2d_name}.hdf5"

shallow_water2d_name = 'ShallowWater2D_subset'
shallow_water2d_path = data_path / f"{shallow_water2d_name}.hdf5"

diffsorp1d_name = 'DiffSorp1D_subset'
diffsorp1d_path = data_path / f"{diffsorp1d_name}.hdf5"

reacdiff2d_name = 'ReacDiff2D_subset'
reacdiff2d_path = data_path / f"{reacdiff2d_name}.hdf5"
