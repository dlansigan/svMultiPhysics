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
    pl_dict['Streamlines'] = pv.Plotter(notebook=False, off_screen=True)
    pl_dict['Streamlines'].open_gif(filepath + f'streamlines.gif')
    clim_dict = {
        'Pressure': [16e3, 17e3],
        'Divergence': [-1e-3, 1e-3],
        'magU': [0, 20],
        'u': [0, 20], #80
        'v': [0, 20],
        'w': [0, 20],
        'magW': [0, 50], # 200
        'magWSS': [0, 20] # 20
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

    # Define streamlines source
    fluid = pv.read(filepath+'geom/fluid.vtp') # Load fluid geometry
    markers = fluid.cell_data["marker"]
    mask = markers == 0
    cell_ids = np.where(mask)[0]
    inlet = fluid.extract_cells(cell_ids) # Extract inlet face
    source_center = inlet.points.mean(axis=0) # Centroid of inlet
    vecs = np.random.normal(size=(100, 3))
    vecs /= np.linalg.norm(vecs, axis=1)[:, None]   # normalize
    radii = 0.8 * np.random.rand(100) ** (1/3)  # scale radius correctly
    source_ball = source_center + vecs * radii[:, None]
    source = pv.PolyData(source_ball)

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

        # Streamlines
        boundary = mesh_def.decimate_boundary().extract_all_edges()
        mesh_def.set_active_vectors('Velocity')
        mesh_def.set_active_scalars('Velocity')
        streamlines = mesh_def.streamlines_from_source(source, vectors="Velocity", integration_direction='forward')
        pl = pl_dict['Streamlines']
        pl.clear() # Clear previous frame
        pl.add_mesh(boundary,opacity=0.3)
        pl.add_mesh(streamlines.tube(radius=0.01),cmap='viridis',clim=clim_dict['magU'])
        pl.add_points(np.array(source_center),color='red')
        pl.camera_position = 'xz'
        pl.add_axes(interactive=True)
        pl.add_text(f"Time step: {filename}", font_size=10)  
        pl.write_frame()
        
    # Close plotters
    for scalar in scalar_names:
        pl_dict[scalar].close()
    pl_dict['Streamlines'].close()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot and save results.')
    parser.add_argument('--path', type=str, help='Path to the case file')
    parser.add_argument('--prefix', type=str, default='4-procs/result', help='Prefix of the result files.')
    args = parser.parse_args()

    # Save animation
    animate_result(args.path, args.prefix)