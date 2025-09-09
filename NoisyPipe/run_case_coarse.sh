case=$1
case_dir=NP_$(printf "%04d" "$case")_coarse

set -x

# Make case directory if it doesn't exist
mkdir -p $case_dir

# Copy template files into case directory
cp solver_temp.xml $case_dir/solver.xml
cp lumen_inlet_temp.flow $case_dir/lumen_inlet.flow

# Update template files with case-specific parameters
# TODO

# Generate geometry and mesh
python generate_noisy_pipe.py --path $case_dir --seed $case --N_ares 32 --N_ref 32 > $case_dir/mesh.log

# Run simulation
cd $case_dir
mpiexec -n 4 svmultiphysics solver.xml > solver.log &
echo "Running job at PID: $!"
