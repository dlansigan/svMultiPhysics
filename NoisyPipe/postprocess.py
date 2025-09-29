import numpy as np
import pyvista as pv
import argparse
import os
import glob
import natsort
from einops import rearrange

def make_bbox(size=(-4.5,4.5,-1.5,1.5,-1.5,1.5), dims=(100,20,20)):
    # Create bounding box
    bbox = pv.Box(size)
    cell_dimensions = np.array(dims)
    return bbox.voxelize(dimensions=cell_dimensions + 1)

def normalize_coords(bbox):
    # Normalize into a cube
    bbox_n = bbox.copy()
    points = bbox.points
    max_val = np.max(points,axis=0)
    min_val = np.min(points,axis=0)
    bbox_n.points = (points-min_val)/(max_val-min_val) # Normalize to (0,1)
    return bbox_n

def get_sdf(bbox,mesh):

    # Compute implicit distance
    bbox_n = bbox.compute_implicit_distance(mesh.extract_surface()) 
    bbox_n = bbox_n.point_data_to_cell_data()

    # Make mask
    bbox_m = bbox_n.copy() 
    bbox_m.cell_data['mask'] = bbox_n.cell_data['implicit_distance']>0 # Positive inside
    bbox_m = bbox_m.point_data_to_cell_data()

    # Get SDF and mask
    sdf = bbox_n.cell_data['implicit_distance'] # Save normalized sdf
    mask = bbox_m.cell_data['mask']

    return sdf, mask

def postproc_data(bbox,mesh,mask):
    # Interpolate results onto query points
    data = bbox.sample(mesh)

    # Get components of velocity
    data['u'] = data['Velocity'][:,0]
    data['v'] = data['Velocity'][:,1]
    data['w'] = data['Velocity'][:,2]

    # Interpolate to cells
    data = data.point_data_to_cell_data()

    # Set values outside of mesh to dummy value
    inverse_mask = [not x for x in mask]
    for scalar in ['u','v','w','Pressure']:
        data.cell_data[scalar][inverse_mask] = 0.0#np.nan

    return data

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Postprocess results for CNF training.')
    # parser.add_argument('--path', type=str, help='Path to the case file')
    parser.add_argument('--start', type=int, help='First case')
    parser.add_argument('--end', type=int, help='Last case')
    parser.add_argument('--prefix', type=str, default='NP', help='Prefix of the case directories.')
    parser.add_argument('--results', type=str, default='4-procs/result', help='Prefix of the result files.')
    args = parser.parse_args()

    # filepath = args.path
    case_start = args.start
    case_end = args.end
    pref = args.prefix
    pref_results = args.results

    # Create directories if they don't exist
    data_path = 'postprocessed_data/' + pref + '_data/'
    os.makedirs(data_path, exist_ok=True)

    # Initialize 
    data = []
    coords = []
    sdfs = []
    sdfs_o = []

    for case in range(case_start,case_end+1):
        print(case)

        # Assemble filepath
        filepath = pref + '/' + pref + '_' + str(case).zfill(4) + '/'
        
        # Check if path exists
        if not os.path.exists(filepath):
            raise ValueError(f"Path {filepath} does not exist.")

        # Get files
        vtu_files = glob.glob(filepath + pref_results+"_*.vtu")
        vtu_files = natsort.natsorted(vtu_files)
        if vtu_files == []:
            raise ValueError(f"No files found with prefix {pref_results} in path {filepath}.")


        if case==case_start:
            # # Read parameters from input file
            # with open(filepath + 'mesh_params.in', 'r') as f:
            #     lines = f.readlines()
            # size = [float(x) for x in lines[28].split(' ')]
            # print(size)
            size = (-4.5,4.5,-1.5,1.5,-1.5,1.5)

            # Get bounding box
            # TODO add code for determining bbox size from mesh params
            # NOTE: move to time loop if deforming/sampling changes?
            bbox = make_bbox(size=size)

            # Normalize box
            bbox_n = normalize_coords(bbox)

            # Get coords
            coords = bbox_n.cell_centers().points
            np.save(data_path + 'coords.npy',coords)
            coords_o = bbox.cell_centers().points
            np.save(data_path + 'coords_o.npy',coords_o)
            # coords_list.append(bbox.cell_centers().points)
            
        # Read mesh from first timestep
        mesh = pv.read(vtu_files[0])

        # Compute SDF, mask
        sdf, mask = get_sdf(bbox_n,mesh)
        sdf_o, mask_o = get_sdf(bbox,mesh)

        data_list = []
        coords_list = []
        for filename in vtu_files:
            # print(filename)

            # Read mesh
            mesh = pv.read(filename)

            # Postprocess data
            data_t = postproc_data(bbox,mesh,mask_o) # Use non-normalized mesh

            # Assemble data array
            data_list.append(np.dstack([data_t.cell_data['u'],
                                data_t.cell_data['v'],
                                data_t.cell_data['w'],
                                data_t.cell_data['Pressure'],
                                # sdf,
                                # mask
                                ]))
            

        # Assemble data array (time N_pts channels)
        data_sim = np.vstack(data_list)
        data.append(data_sim)
        sdfs.append(sdf)
        sdfs_o.append(sdf_o)
        # np.save(data_path + pref + '_' + str(case).zfill(4) + '_data.npy', data_sim)

    # Assemble coords array (N_pts components)
    # coords = np.stack(coords_list)
    # coords.append(coords_sim)
    # np.save(data_path + pref + '_' + str(case).zfill(4) + '_coords.npy', coords_sim)

    # Assemble full arrays 
    data = np.vstack(data) # (N_samp N_pts channels)
    sdfs = np.stack(sdfs) # (N_geom N_pts)
    sdfs = np.reshape(sdfs,(case_end-case_start+1,-1,1))
    sdfs_o = np.stack(sdfs_o) # (N_geom N_pts)
    sdfs_o = np.reshape(sdfs_o,(case_end-case_start+1,-1,1))
    # coords = np.vstack(coords) # (N_samp N components)

    # Save
    # np.save(data_path + 'coords.npy',coords)
    np.save(data_path + 'data.npy',data)
    np.save(data_path + 'sdf.npy',sdfs)
    np.save(data_path + 'sdf_o.npy',sdfs_o)
