import tetgen
import pyvista as pv
from vtkmodules.vtkFiltersModeling import vtkRuledSurfaceFilter
from vtkmodules.vtkFiltersModeling import vtkFillHolesFilter
from vtk import vtkAppendPolyData
import numpy as np
from scipy.interpolate import make_interp_spline
import argparse
import os

def generate_geometry(radius, h, t, **kwargs):
    # Unpack kwargs
    num_modes = kwargs.get("num_modes", 0)
    kmin = kwargs.get("kmin", 2)
    kmax = kwargs.get("kmax", 8)
    Amin = kwargs.get("Amin", -0.1)
    Amax = kwargs.get("Amax", 0.1)
    N_ares = kwargs.get("N_ares",64)
    N_base = kwargs.get("N_base", 32)
    N_xres = kwargs.get("N_xres", 100)
    N_layers = kwargs.get("N_layers", 0)
    bump_w = kwargs.get("bump_w", 0.0)
    bump_h = kwargs.get("bump_h", 0.0)
    bump_x = kwargs.get("bump_x", 0.0)
    random_seed = kwargs.get("random_seed", None)

    # Initialize
    contours = [] # Inner

    # x locations
    xpos = np.linspace(-h/2,h/2,N_base)
    x_ref = np.linspace(-h/2,h/2,N_xres)

    # Define radius as function of x
    r_x = (radius - bump_h * (radius + np.cos(np.pi * ((xpos-bump_x) / bump_w))))
    r_x[np.abs(xpos-bump_x) > bump_w] = radius

    # Angle locations
    theta = np.linspace(0, 2 * np.pi, N_ares, endpoint=False)

    # Random seed, if defined
    if random_seed is not None:
        np.random.seed(random_seed)

    ##### Noisy Contours #####

    # Generate base contours with noise
    for i in range(N_base):
        r = r_x[i]
        xloc = xpos[i]

        for j in range(num_modes):
            # Get random number
            k = np.random.randint(kmin,kmax)
            A = np.random.uniform(Amin,Amax)

            # Define radial function r(θ)
            r += A * np.cos(k*theta) # Multimodal noise

        # Convert to Cartesian (x, y)
        y,z = r * np.cos(theta), r * np.sin(theta)

        # x positions for shape
        x = np.full_like(y, xloc)

        # Stack points
        slice_points = np.column_stack([x,y,z])
        
        # Add to contours PolyData
        contours.append(slice_points)

    # Spline between contours to make surface smoother
    b_spline = []
    for i in range(N_ares): # First, get splines for each theta
        r_x = np.zeros(N_base)
        for j in range(N_base):
            contour_point = contours[j][i,:] # Get point at contour j at theta_i
            r_x[j] = np.sqrt(contour_point[1]**2 + contour_point[2]**2)
        b_spline.append(make_interp_spline(xpos, r_x))
    points_list = []
    points_list_o = []
    for i in range(N_xres): # Then, get new contours at refined x locations
        xloc = x_ref[i] 

        # Evaluate splines at xloc
        r = np.zeros(N_ares)
        for j in range(N_ares):
            r[j] = b_spline[j](xloc)

        # Convert to Cartesian (x, y)
        y,z = r * np.cos(theta), r * np.sin(theta)
        yo,zo = (r+t) * np.cos(theta), (r+t) * np.sin(theta)

        # x positions for shape
        x = np.full_like(y, xloc)
        xo = np.full_like(yo, xloc)

        # Stack points
        slice_points = np.column_stack([x,y,z])
        slice_points_o = np.column_stack([xo,yo,zo])

        # Store new contours
        points_list.append(slice_points)
        points_list_o.append(slice_points_o)
        
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
        fluid = wall.merge(inlet).merge(outlet)

    ##### Solid Geometry #####

    # Solid wall
    wall_o = build_wall(points_list_o)#pv.PolyData(points_o, faces_o)
    # wall_o = wall_o.triangulate()

    # Solid wall inlet/outlet 
    mid_points = (points_list_o[-1] + points_list[-1])/2 # Identify a contour in the solid region for labeling later
    region_solid = mid_points[0] # Only need one
    solid_inlet = build_annulus(points_list[-1],points_list_o[-1])
    solid_outlet = build_annulus(points_list[0],points_list_o[0])

    # Final solid geometry
    solid = solid_inlet.merge(solid_outlet).merge(wall_o).merge(wall)

    ##### Finalize #####

    # Label patches 
    inlet.cell_data['marker']=0
    outlet.cell_data['marker']=1
    wall.cell_data['marker']=2
    wall_o.cell_data['marker']=3
    solid_inlet.cell_data['marker']=4
    solid_outlet.cell_data['marker']=5

    # Merge everything
    regions = [region_solid]
    # regions = []
    if N_layers > 0:
        combined = pv.merge(layers)
        combined = combined.merge(solid_inlet).merge(solid_outlet).merge(wall_o)
        regions.extend(region_bl)
    else:
        combined = solid_inlet.merge(solid_outlet).merge(wall_o).merge(wall).merge(inlet).merge(outlet)
        # combined = wall.merge(inlet).merge(outlet)

    np.save(geom_path + "region.npy", regions) # DEBUG

    return fluid, solid, combined, regions

def build_wall(points_list):   

    # Get sizes
    N_x = len(points_list)
    N_a = points_list[0].shape[0]

    # Combine all points
    points = np.vstack(points_list)

    # Create quad faces for the walls
    faces = []
    for layer in range(N_x - 1):  # loop over layers
        offset0 = layer * N_a
        offset1 = (layer + 1) * N_a
        for i in range(N_ares):
            i0 = offset0 + i
            i1 = offset0 + (i + 1) % N_a
            j1 = offset1 + (i + 1) % N_a
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
    for j in range(N_ares):
        # Indices
        a0 = j
        a1 = (j + 1) % N_ares
        b0 = j + N_ares
        b1 = ((j + 1) % N_ares) + N_ares

        # Triangles for quad face between (a0, a1, b1, b0)
        faces.append([3, a0, a1, b1])
        faces.append([3, a0, b1, b0])

    faces = np.array(faces, dtype=np.int32).flatten()
    surf = pv.PolyData(points,faces)
    return surf

def build_cap(points):
    N_a = points.shape[0] # Get angular resolution
    cap = pv.Polygon(center=(0,0,-0.5), radius=1.0, n_sides=N_a) # Initialize base polygon
    cap.points = points # Replace initial points with actual points
    return cap

def build_boundary_layer(points_list, initial_height, n_layers, growth_ratio):

    # Initialize stuff
    layers = []
    region_bl = []
    N_x = len(points_list)
    N_a = points_list[0].shape[0]
    theta = np.linspace(0, 2 * np.pi, N_a, endpoint=False)

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
            for k in range(N_a):
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
    for pt, marker, max_vol in region_points:
        tet.add_region(marker, pt)#, max_vol=max_vol)

    # Tetrahedralize with quality and volume constraints 
    _, _, attrib = tet.tetrahedralize(switches=switches,verbose=True)

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

def split_surfaces(mesh):

    # Get the surface from mesh
    surf = mesh.extract_surface(pass_cellid=True,pass_pointid=True)

    # Isolate inlet/outlet/walls
    surf_n = surf.compute_normals(point_normals=True, cell_normals=True, inplace=False)
    normals = surf_n.cell_data['Normals']
    dot_threshold = 0.999  # cosine of angle (close to 1 = nearly aligned)
    axis = np.array([1.0, 0.0, 0.0]) 
    dot = normals @ axis
    inlet_ids = np.where(dot < -dot_threshold)[0]
    outlet_ids = np.where(dot > dot_threshold)[0]
    wall_ids = np.where(np.abs(dot) <= dot_threshold)[0]
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
    radius = float(lines[4])
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
    N_ares = int(lines[26])
    N_base = int(lines[28])
    N_xres = int(lines[30])
    N_layers = int(lines[32]) 
    growth_ratio = float(lines[34])
    init_height = float(lines[36])
    max_vols = lines[38].split(' ')
    seed = int(lines[40])
    switches = lines[42]

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
    fluid, solid, combined, regions = generate_geometry(
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
        N_ares=N_ares,
        N_base=N_base,
        N_xres=N_xres,
        N_layers=N_layers,
        random_seed=seed
    )
    print('-- Fluid manifold?', fluid.is_manifold)
    print('-- Solid manifold?', solid.is_manifold)
    print("Done generating geometry.")

    # Save geometries for visualization later
    fluid.save(geom_path + "fluid.vtp")
    solid.save(geom_path + "solid.vtp")
    combined.save(geom_path + "combined.vtp")
    # np.save(geom_path + "region_solid.npy", region_solid)
    print("Saved geometry files.")
    print()

    # Define region seed points
    region_points = [
        ([0, 0, 0], 0, float(max_vols[0])),  # inside fluid
    ]
    for i,pts in enumerate(regions):
        region_points.append((pts, i+1, float(max_vols[i+1])))

    # Generate mesh
    mesh, attrib = mesh_geometry(combined,region_points,switches=switches)
    print(f"Generated mesh with {mesh.n_points} nodes and {mesh.n_cells} elements.")

    # Extract fluid and solid meshes
    markers = [[0]+list(range(2,len(region_points))),[1]]
    meshes = extract_region(mesh,markers,attrib)
    fluid_mesh = meshes[0]
    # if N_layers > 0: # Make sure all regions included in fluid mesh
    #     fluid_mesh = meshes[0]
    solid_mesh = meshes[1]

    # Split surfaces
    inlet_surf_f, outlet_surf_f, wall_surfs_f = split_surfaces(fluid_mesh)
    inlet_surf_s, outlet_surf_s, wall_surfs_s = split_surfaces(solid_mesh)

    # Save the volume and surface meshes
    fluid_mesh.save(mesh_path + "fluid/mesh-complete.vtu")
    inlet_surf_f.extract_surface(pass_cellid=True,pass_pointid=True).save(mesh_path + "fluid/mesh-surfaces/inlet.vtp")
    outlet_surf_f.extract_surface(pass_cellid=True,pass_pointid=True).save(mesh_path + "fluid/mesh-surfaces/outlet.vtp")
    wall_surfs_f[0].extract_surface(pass_cellid=True,pass_pointid=True).save(mesh_path + "fluid/mesh-surfaces/interface.vtp")
    solid_mesh.save(mesh_path + "solid/mesh-complete.vtu")
    inlet_surf_s.extract_surface(pass_cellid=True,pass_pointid=True).save(mesh_path + "solid/mesh-surfaces/inlet.vtp")
    outlet_surf_s.extract_surface(pass_cellid=True,pass_pointid=True).save(mesh_path + "solid/mesh-surfaces/outlet.vtp")

    # Select which wall is inside vs outside based on size
    if wall_surfs_s[0].area < wall_surfs_s[1].area:
        wall_surfs_s = (wall_surfs_s[0], wall_surfs_s[1])
    else:
        wall_surfs_s = (wall_surfs_s[1], wall_surfs_s[0])
    wall_surfs_s[1].extract_surface(pass_cellid=True,pass_pointid=True).save(mesh_path + "solid/mesh-surfaces/outside.vtp")
    wall_surfs_s[0].extract_surface(pass_cellid=True,pass_pointid=True).save(mesh_path + "solid/mesh-surfaces/interface.vtp")
    # wall_surfs_f[0].extract_surface(pass_cellid=True,pass_pointid=True).save(mesh_path + "solid/mesh-surfaces/interface.vtp") # Should be the same
    print("Saved mesh files. Done!")
    print()
