Darcy_2d_betta_01_dict = {
                         'equation': 'Steady-state Darcy flow: du/dt - div(a(x) * grad(u))',
                         'right_part': 0.1,
                         'definition_area': 'x in [0,1]^2',
                         'boundary_condition': 'Dirichlet condition',
                         'init_condition': 'random init condition',
                         'inputs_var': """x (np.ndarray): 1D array of x-coordinates, shape (Nx,)
                                          y (np.ndarray): 1D array of y-coordinates, shape (Ny,)
                                          a (np.ndarray): 2D coefficient array, shape (Nx, Ny)""",
                         'outputs_var': 'u np.array shape (Nx, Ny)'
                        }

ReacDiff_Nu05_Rho20_dict = {
                            'equation': 'du/dt - 0.5 * d^2u/dx^2 - 2 * u * (1-u)',
                            'right_part': 0.0,
                            'definition_area': 'x in [0,1], t in (0,1]',
                            'boundary_condition': 'periodic',
                            'init_condition': 'u(0, x) superposition of sinusoidal waves, normalized and abs applied',
                            'inputs_var': 'x np.array with shape (Nx,), t np.array with shape (Nt,) and u(0, x) np.array with shape (Nx,)',
                            'outputs_var': 'u np.array with shape (Nt, Nx)'
                            }

