import tetgen
import pyvista as pv
from vtkmodules.vtkFiltersModeling import vtkRuledSurfaceFilter
from vtkmodules.vtkFiltersModeling import vtkFillHolesFilter
from vtk import vtkAppendPolyData
import numpy as np
from scipy.interpolate import make_interp_spline
import argparse
import os

def generate_geometry(radius, spline_bounds, t, **kwargs):
    # Unpack kwargs
    num_modes = kwargs.get("num_modes", 0)
    kmin = kwargs.get("kmin", 2)
    kmax = kwargs.get("kmax", 8)
    Amin = kwargs.get("Amin", -0.1)
    Amax = kwargs.get("Amax", 0.1)
    N_theta = kwargs.get("N_theta",64)
    N_base = kwargs.get("N_base", 32)
    N_x = kwargs.get("N_x", 100)
    N_spline = kwargs.get("N_spline", 5)
    N_layers = kwargs.get("N_layers", 0)
    bump_w = kwargs.get("bump_w", 0.0)
    bump_h_std = kwargs.get("bump_h", 0.0)
    bump_x = kwargs.get("bump_x", 0.0)
    random_seed = kwargs.get("random_seed", None)

    # Initialize
    contours = [] # Inner

    # x locations
    xpos = np.linspace(spline_bounds[0],spline_bounds[1],N_base)

    # Define radius as function of x
    bump_h = np.random.normal(0.0,bump_h_std/2)
    print(bump_h,bump_w)
    radius = np.random.uniform(radius[0],radius[1])
    r_x = (radius - bump_h * (radius + np.cos(np.pi * ((xpos-bump_x) / bump_w))))
    r_x[np.abs(xpos-bump_x) > bump_w] = radius

    # Angle locations
    theta = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)

    # Random seed, if defined
    if random_seed is not None:
        np.random.seed(random_seed)

    ##### Noisy Contours #####

    # Make spline within 3D box using random points
    x_spline = np.linspace(spline_bounds[0],spline_bounds[1],N_spline)
    x_spline = x_spline.reshape(N_spline,1)
    y_spline = np.random.uniform(spline_bounds[2],spline_bounds[3],N_spline)
    y_spline = y_spline.reshape(N_spline,1)
    z_spline = np.random.uniform(spline_bounds[4],spline_bounds[5],N_spline)
    z_spline = z_spline.reshape(N_spline,1)
    s_points = np.hstack((x_spline,y_spline,z_spline))
    spline = pv.Spline(s_points, N_x)
    spline = spline.points
    np.save(geom_path + "spline.npy", spline)

    # Get orthonormal axes
    T, N, B = compute_rmf_frames(spline)

    # Generate base contours with noise
    for i in range(N_base):
        # Get radius at this point
        r = r_x[i]
        # r = np.random.uniform(radius[0],radius[1])

        # Find spline points at this location
        ii = i*N_x//N_base
        s = spline[ii,:]

        for j in range(num_modes):
            # Get random number
            k = np.random.randint(kmin,kmax)
            A = np.random.uniform(Amin,Amax)

            # Define radial function r(θ)
            r += A * np.cos(k*theta) # Multimodal noise

        # Convert to Cartesian (x, y)
        x= s[0] + r * (np.cos(theta) * N[ii,0] + np.sin(theta) * B[ii,0])
        y= s[1] + r * (np.cos(theta) * N[ii,1] + np.sin(theta) * B[ii,1])
        z= s[2] + r * (np.cos(theta) * N[ii,2] + np.sin(theta) * B[ii,2])

        # Stack points
        slice_points = np.column_stack([x,y,z])
        
        # Add to contours PolyData
        contours.append(slice_points)
    np.save(geom_path+'contours.npy',np.array(contours))

    # Spline between contours to make surface smoother
    b_spline = []
    for i in range(N_theta): # First, get splines for each theta
        s = np.zeros((N_base,3))
        for j in range(N_base):
            contour_point = contours[j][i,:] # Get point at contour j at theta_i
            s[j,:] = contour_point
        b_spline.append(pv.Spline(s,N_x).points)
    points_list = []
    for i in range(N_x): # Then, get new contours at refined x locations

        # Evaluate splines at xloc
        r = np.zeros(N_theta)
        x = np.zeros(N_theta)
        y = np.zeros(N_theta)
        z = np.zeros(N_theta)
        for j in range(N_theta):
            x[j],y[j],z[j] = b_spline[j][i,:]

        # Stack points
        slice_points = np.column_stack([x,y,z])

        # Store new contours
        points_list.append(slice_points)
        
    ##### Fluid Geometry #####

    # Build boundary layer if defined
    if N_layers > 0:
        print("Building boundary layer...")
        layers, inlet, outlet, wall, region_bl = build_boundary_layer(
            points_list,
            initial_height=init_height,
            n_layers=N_layers,
            growth_ratio=growth_ratio
            )
        print(f"Done building boundary layer.")
        print()
        fluid = pv.merge(layers)
    # If no boundary layer, just build fluid geometry
    else:
        # Fluid wall (interface, if FSI)
        wall = build_wall(points_list)
        # wall = wall.triangulate()

        # Fluid inlet
        inlet = build_cap(points_list[0])
        # inlet = inlet.triangulate()

        # Fluid outlet
        outlet = build_cap(points_list[-1])
        # outlet = outlet.triangulate()

        # Final fluid geometry
        # fluid = wall.merge(inlet).merge(outlet)

    ##### Finalize #####

    # Label patches 
    inlet.cell_data['marker']=0
    outlet.cell_data['marker']=1
    wall.cell_data['marker']=2
    fluid = wall.merge(inlet).merge(outlet)

    # Get inlet/outlet axes
    in_axis = T[0,:]
    out_axis = T[-1,:]

    return fluid, in_axis, out_axis

def compute_rmf_frames(points, initial_normal=None):
    """
    Compute rotation-minimizing frames (RMF) along a spline
    using parallel transport.

    points: (N, 3) array of 3D spline points
    initial_normal: optional (3,) vector to seed the normal

    Returns:
        T: (N, 3) tangent vectors
        N: (N, 3) normal vectors
        B: (N, 3) binormal vectors
    """
    N_pts = len(points)
    T = np.gradient(points, axis=0)
    T /= np.linalg.norm(T, axis=1)[:, None]

    N = np.zeros_like(T)
    B = np.zeros_like(T)

    # Step 1: Choose consistent initial normal
    if initial_normal is None:
        up = np.array([0, 0, 1])
        if np.allclose(T[0], up):
            up = np.array([1, 0, 0])
        n0 = np.cross(T[0], up)
        n0 /= np.linalg.norm(n0)
    else:
        n0 = initial_normal / np.linalg.norm(initial_normal)

    N[0] = n0
    B[0] = np.cross(T[0], N[0])

    # Step 2: Parallel transport
    for i in range(1, N_pts):
        v = T[i-1]
        w = T[i]
        axis = np.cross(v, w)
        axis_len = np.linalg.norm(axis)

        if axis_len < 1e-6:
            # No significant rotation — keep same frame
            N[i] = N[i-1]
        else:
            axis /= axis_len
            angle = np.arccos(np.clip(np.dot(v, w), -1.0, 1.0))
            # Rodrigues rotation
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            N[i] = R @ N[i-1]

        N[i] /= np.linalg.norm(N[i])
        B[i] = np.cross(T[i], N[i])
        B[i] /= np.linalg.norm(B[i])

    return T, N, B


def build_wall(points_list):   

    # Get sizes
    N_x = len(points_list)
    N_theta = points_list[0].shape[0]

    # Combine all points
    points = np.vstack(points_list)

    # Create quad faces for the walls
    faces = []
    for layer in range(N_x - 1):  # loop over layers
        offset0 = layer * N_theta
        offset1 = (layer + 1) * N_theta
        for i in range(N_theta):
            i0 = offset0 + i
            i1 = offset0 + (i + 1) % N_theta
            j1 = offset1 + (i + 1) % N_theta
            j0 = offset1 + i
            faces.append([4, i0, i1, j1, j0])
            
    # Flatten face list
    faces = np.hstack(faces)
    wall = pv.PolyData(points, faces)
    return wall

def build_annulus(points_list_i, points_list_o):

    # Combine points on outer and inner contours of annulus
    points = np.vstack([points_list_o,points_list_i])

    # Create faces for the annuli
    faces = []
    for j in range(N_theta):
        # Indices
        a0 = j
        a1 = (j + 1) % N_theta
        b0 = j + N_theta
        b1 = ((j + 1) % N_theta) + N_theta

        # Triangles for quad face between (a0, a1, b1, b0)
        faces.append([3, a0, a1, b1])
        faces.append([3, a0, b1, b0])

    faces = np.array(faces, dtype=np.int32).flatten()
    surf = pv.PolyData(points,faces)
    return surf

def build_cap(points):
    N_theta = points.shape[0] # Get angular resolution
    cap = pv.Polygon(center=(0,0,-0.5), radius=1.0, n_sides=N_theta) # Initialize base polygon
    cap.points = points # Replace initial points with actual points
    return cap

def build_boundary_layer(points_list, initial_height, n_layers, growth_ratio):

    # Initialize stuff
    layers = []
    region_bl = []
    N_x = len(points_list)
    N_theta = points_list[0].shape[0]
    theta = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)

    # Create the boundary layer
    for i in range(n_layers):
        # Calculate the height of the current layer
        layer_height = initial_height * (growth_ratio ** i)

        # Get contour at x location
        points_new_list = []
        for j in range(N_x):
            points = points_list[j]
            points_new = np.empty_like(points)
            # Determine new points at each theta
            for k in range(N_theta):
                r = np.sqrt(points[k,1]**2+points[k,2]**2)
                r_new = r - layer_height # Offset inward
                points_new[k,0] = points[k,0]
                points_new[k,1] = r_new * np.cos(theta[k])
                points_new[k,2] = r_new * np.sin(theta[k])
            points_new_list.append(points_new)

        # Create wall faces
        wall = build_wall(points_new_list)

        # Store a point between previous and current layer for region points
        mid_points = (points_list[-1]+points_new_list[-1])/2
        region_bl.append(mid_points[0]) # Just need one 

        # Create annuli
        inlet = build_annulus(points_new_list[-1],points_list[-1])
        outlet = build_annulus(points_new_list[0],points_list[0])

        # Label patches 
        inlet.cell_data['marker']=6+3*i
        outlet.cell_data['marker']=7+3*i
        wall.cell_data['marker']=8+3*i

        # Merge surfaces into one
        layer = wall.merge(inlet).merge(outlet)

        # Store surfaces to list
        layers.append(layer)

        # New surface becomes old surface
        points_list = points_new_list

    # Create inner fluid
    inlet = build_cap(points_list[0])
    outlet = build_cap(points_list[-1])
    layer = inlet.merge(outlet)
    layers.append(layer)

    return layers, inlet, outlet, wall, region_bl

def mesh_geometry(geo,region_points,switches='pzq1.2Aa0.1f'):
    # Get surface and triangulate
    surf = geo.extract_surface().triangulate().clean()
    
    # Create TetGen object
    tet = tetgen.TetGen(surf)

    # Add regions
    # for pt, marker, max_vol in region_points:
    #     tet.add_region(marker, pt)#, max_vol=max_vol)

    # Tetrahedralize with quality and volume constraints 
    try:
        _, _, attrib = tet.tetrahedralize(switches=switches,verbose=True)
        print('Meshing successful.')
    except Exception as e:
        print('An error occured in meshing.')
        return None, None

    # Assign global IDs
    mesh = tet.grid
    mesh.cell_data["GlobalElementID"] = np.arange(mesh.n_cells, dtype=np.int32)  
    mesh.point_data["GlobalNodeID"] = np.arange(mesh.n_points, dtype=np.int32)

    return mesh, attrib

def extract_region(mesh,markers,attrib):
    meshes = []
    for marker in markers:
        # reg_mask = attrib[:,0] == marker
        reg_mask = np.logical_or.reduce([attrib[:,0] == value for value in marker])
        region = mesh.extract_cells(reg_mask)
        meshes.append(region)
    return meshes

def split_surfaces(mesh, in_axis=np.array([1.0, 0.0, 0.0]) , out_axis=np.array([1.0, 0.0, 0.0]) ):

    # Get the surface from mesh
    surf = mesh.extract_surface(pass_cellid=True,pass_pointid=True)

    # Get normals
    surf_n = surf.compute_normals(point_normals=True, cell_normals=True, inplace=False)
    normals = surf_n.cell_data['Normals']

    # Define box regions near each end along the pipe axis 
    x_min, x_max, _,_,_,_ = surf_n.bounds
    x_coords = surf_n.cell_centers().points[:, 0]  # cell center x-coordinates
    inlet_box = (x_coords <= x_min+0.2*(x_max-x_min))
    outlet_box = (x_coords >= x_max-0.2*(x_max-x_min)) 

    # Take dot product to assess normal alignment with in/out axes
    dot_threshold = 0.999  # cosine of angle (close to 1 = nearly aligned)
    in_dot = np.abs(normals @ in_axis)
    out_dot = np.abs(normals @ out_axis)

    # Inlet/Outlet
    inlet_ids = np.where((in_dot > dot_threshold) & inlet_box)[0]
    outlet_ids = np.where((out_dot > dot_threshold) & outlet_box)[0]

    # Walls (everything else)
    wall_ids = np.setdiff1d(np.arange(surf_n.n_cells), np.concatenate([inlet_ids, outlet_ids]))
    
    # Split surfaces
    inlet_surf = surf_n.extract_cells(inlet_ids)
    outlet_surf = surf_n.extract_cells(outlet_ids)
    wall_surf = surf_n.extract_cells(wall_ids)

    # Separate walls if multiple (e.g., inner and outer walls)
    wall_surfs = wall_surf.connectivity().split_bodies()
    return inlet_surf, outlet_surf, wall_surfs

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate and mesh a noisy pipe geometry.')
    parser.add_argument('-i', '--input', required=True, type=str, help='Filepath to parameters input file.')
    args = parser.parse_args()

    # Read parameters from input file
    with open(args.input, 'r') as f:
        lines = f.readlines()
    radius = [float(x) for x in lines[4].split(' ')]
    height = float(lines[6])
    thickness = float(lines[8])
    num_modes = int(lines[10])
    kmin = int(lines[12])
    kmax = int(lines[14])
    Amin = float(lines[16])
    Amax = float(lines[18])
    bump_w = float(lines[20])
    bump_h = float(lines[22])
    bump_x = float(lines[24])
    N_spline = int(lines[26])
    spline_bounds = [float(x) for x in lines[28].split(' ')]
    N_theta = int(lines[30])
    N_base = int(lines[32])
    N_x = int(lines[34])
    N_layers = int(lines[36]) 
    growth_ratio = float(lines[38])
    init_height = float(lines[40])
    max_vols = [float(x) for x in lines[42].split(' ')]
    seed = int(lines[44])
    switches = lines[46]

    # Define geometry and mesh path (assume we are in case directory)
    mesh_path = "mesh/"
    geom_path = "geom/"

    # Create directories if they don't exist
    os.makedirs(mesh_path, exist_ok=True)
    os.makedirs(mesh_path + "fluid/mesh-surfaces/", exist_ok=True)
    os.makedirs(mesh_path + "solid/mesh-surfaces/", exist_ok=True)
    os.makedirs(geom_path, exist_ok=True)

    # Generate geometry
    print("Generating geometry...")
    fluid, in_axis, out_axis = generate_geometry(
        radius=radius,
        h=height,
        t=thickness,
        num_modes=num_modes,
        kmin=kmin,
        kmax=kmax,
        Amin=Amin,
        Amax=Amax,
        bump_x=bump_x,
        bump_w=bump_w,
        bump_h=bump_h,
        N_spline=N_spline,
        spline_bounds=spline_bounds,
        N_theta=N_theta,
        N_base=N_base,
        N_x=N_x,
        N_layers=N_layers,
        random_seed=seed
    )
    print('-- Fluid manifold?', fluid.is_manifold)
    print("Done generating geometry.")

    # Save geometries for visualization later
    fluid.save(geom_path + "fluid.vtp")
    # np.save(geom_path + "region_solid.npy", region_solid)
    print("Saved geometry files.")
    print()

    # # Define region seed points
    # region_points = [
    #     ([0, 0, 0], 0, float(max_vols[0])),  # inside fluid
    # ]
    # for i,pts in enumerate(regions):
    #     region_points.append((pts, i+1, float(max_vols[i+1])))

    # Generate mesh
    mesh, attrib = mesh_geometry(fluid,[],switches=switches)
    print(f"Generated mesh with {mesh.n_points} nodes and {mesh.n_cells} elements.")

    # # Extract fluid and solid meshes
    # markers = [[0]+list(range(2,len(region_points))),[1]]
    # meshes = extract_region(mesh,markers,attrib)
    # fluid_mesh = meshes[0]
    # # if N_layers > 0: # Make sure all regions included in fluid mesh
    # #     fluid_mesh = meshes[0]
    # solid_mesh = meshes[1]

    # Split surfaces
    fluid_mesh = mesh
    inlet_surf_f, outlet_surf_f, wall_surfs_f = split_surfaces(fluid_mesh,in_axis=in_axis,out_axis=out_axis)
    # inlet_surf_s, outlet_surf_s, wall_surfs_s = split_surfaces(solid_mesh)

    # # Save the volume and surface meshes
    fluid_mesh.save(mesh_path + "fluid/mesh-complete.vtu")
    inlet_surf_f.extract_surface(pass_cellid=True,pass_pointid=True).save(mesh_path + "fluid/mesh-surfaces/inlet.vtp")
    outlet_surf_f.extract_surface(pass_cellid=True,pass_pointid=True).save(mesh_path + "fluid/mesh-surfaces/outlet.vtp")
    wall_surfs_f[0].extract_surface(pass_cellid=True,pass_pointid=True).save(mesh_path + "fluid/mesh-surfaces/interface.vtp")
    # solid_mesh.save(mesh_path + "solid/mesh-complete.vtu")
    # inlet_surf_s.extract_surface(pass_cellid=True,pass_pointid=True).save(mesh_path + "solid/mesh-surfaces/inlet.vtp")
    # outlet_surf_s.extract_surface(pass_cellid=True,pass_pointid=True).save(mesh_path + "solid/mesh-surfaces/outlet.vtp")

    # # Select which wall is inside vs outside based on size
    # if wall_surfs_s[0].area < wall_surfs_s[1].area:
    #     wall_surfs_s = (wall_surfs_s[0], wall_surfs_s[1])
    # else:
    #     wall_surfs_s = (wall_surfs_s[1], wall_surfs_s[0])
    # wall_surfs_s[1].extract_surface(pass_cellid=True,pass_pointid=True).save(mesh_path + "solid/mesh-surfaces/outside.vtp")
    # wall_surfs_s[0].extract_surface(pass_cellid=True,pass_pointid=True).save(mesh_path + "solid/mesh-surfaces/interface.vtp")
    # # wall_surfs_f[0].extract_surface(pass_cellid=True,pass_pointid=True).save(mesh_path + "solid/mesh-surfaces/interface.vtp") # Should be the same
    print("Saved mesh files. Done!")
    print()
