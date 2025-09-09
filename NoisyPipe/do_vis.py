import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import glob
import natsort
import os
import argparse

def animate_result(filepath, pref, cam='xz'):
    # Check if path exists
    if not filepath.endswith('/'):
        filepath += '/'
    if not os.path.exists(filepath):
        raise ValueError(f"Path {filepath} does not exist.")

    # Get files
    vtu_files = glob.glob(filepath + pref+"_*.vtu")
    vtu_files = natsort.natsorted(vtu_files)
    if vtu_files == []:
        raise ValueError(f"No files found with prefix {pref} in path {filepath}.")
    
    # Define scalars
    scalar_names = ['Pressure', 'Divergence', 'magU', 'u', 'v', 'w', 'magW', 'magWSS']

    # Initialize plotters
    pl_dict = {}
    for scalar in scalar_names:
        # Create a new plotter for each scalar, store in dictionary
        pl_dict[scalar] = pv.Plotter(notebook=False, off_screen=True)
        pl_dict[scalar].open_gif(filepath + f'{scalar}.gif')
    clim_dict = {
        'Pressure': [0, 3e5],
        'Divergence': [-1e-3, 1e-3],
        'magU': [0, 300],
        'u': [0, 100], #80
        'v': [0, 100],
        'w': [0, 100],
        'magW': [0, 5000], # 200
        'magWSS': [0, 100] # 20
    }
    clip_dict = {
        'Pressure': True,
        'Divergence': True,
        'magU': True,
        'u': True,
        'v': True,
        'w': True,
        'magW': True,
        'magWSS': False
    }

    # Make and save frames to gif
    for filename in vtu_files:
        # Print timestep
        print(filename)

        # Read mesh
        mesh = pv.read(filename)
        
        # Get displaced mesh, if FSI
        mesh_def = mesh.copy()
        # delta = mesh.point_data['Displacement']
        # mesh_def.points = mesh.points + delta

        # Make scalars

        # Pressure
        mesh_def['Pressure'] = mesh_def.point_data['Pressure']

        # Divergence
        mesh_def['Divergence'] = mesh_def.point_data['Divergence']

        # Magnitude of velocity
        u = mesh_def.point_data['Velocity']
        magU = np.linalg.norm(u,axis=1)
        mesh_def['magU'] = magU

        # u, v, w
        mesh_def['u'] = u[:,0]
        mesh_def['v'] = u[:,1]
        mesh_def['w'] = u[:,2]

        # Magnitude of vorticity
        w = mesh_def.point_data['Vorticity']
        magW = np.linalg.norm(w,axis=1)
        mesh_def['magW'] = magW

        # Magnitude of WSS
        WSS = mesh_def.point_data['WSS']
        magWSS = np.linalg.norm(WSS,axis=1)
        mesh_def['magWSS'] = magWSS

        # Plot

        for scalar in scalar_names:
            pl = pl_dict[scalar]
            pl.clear() # Clear previous frame
            # Show only half depending on scalar plotted
            if clip_dict[scalar]:
                half_pipe = mesh_def.clip(normal='y',origin=mesh.center,invert=False)
            else:
                half_pipe = mesh_def
            pl.add_mesh(half_pipe, scalars=scalar, cmap="viridis", show_edges=False, clim=clim_dict[scalar])
            pl.camera_position = cam
            pl.add_axes(interactive=True)
            pl.add_text(f"Time step: {filename}", font_size=10)  
            pl.write_frame()

    # Close plotters
    for scalar in scalar_names:
        pl_dict[scalar].close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot and save results.')
    parser.add_argument('--path', type=str, help='Path to the case file')
    parser.add_argument('--prefix', type=str, default='4-procs/result', help='Prefix of the result files.')
    args = parser.parse_args()

    # Save animation
    animate_result(args.path, args.prefix)