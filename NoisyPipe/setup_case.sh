case=$1
case_name=$2
input_name=$3
DT=$4
NT=$5
DUMP_FREQ=$6

# Set simulation home directory
SIM_HOME=/home/dana/Documents/SimVascular/svMultiPhysics/NoisyPipe

# Set case directory to case name if provided, else use default naming
if [ -z "$case_name" ]; then
    case_dir=SP/SP_$(printf "%04d" "$case")
else
    case_dir=$case_name
fi

# Set DT, NT, DUMP_FREQ to default values if not provided
if [ -z "$DT" ]; then
    DT=0.005
fi
if [ -z "$NT" ]; then
    NT=200
fi
if [ -z "$DUMP_FREQ" ]; then
    DUMP_FREQ=2
fi

# set -x

# Make case directory if it doesn't exist
mkdir -p $case_dir

# Copy template files into case directory
cp solver_temp.xml $case_dir/solver.xml
cp lumen_inlet_temp.flow $case_dir/lumen_inlet.flow
if [ -z "$input_name" ]; then
    mesh_input=mesh_params_temp.in
else
    mesh_input=${input_name}
fi
cp $mesh_input $case_dir/mesh_params.in
echo "Copied template files:"
echo "--solver_temp.xml -> $case_dir/solver.xml"
echo "--lumen_inlet_temp.flow -> $case_dir/lumen_inlet.flow"
echo "--$mesh_input -> $case_dir/mesh_params.in"

# Update template files with case-specific parameters
sed -i "s/<NT>/$NT/g" $case_dir/solver.xml
sed -i "s/<DT>/$DT/g" $case_dir/solver.xml
sed -i "s/<DUMP_FREQ>/$DUMP_FREQ/g" $case_dir/solver.xml
sed -i "s/<SEED>/$case/g" $case_dir/mesh_params.in

# Generate geometry and mesh
cd $case_dir
python $SIM_HOME/generate_noisy_pipe.py --input mesh_params.in > mesh.log
