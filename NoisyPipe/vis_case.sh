case=$1
case_name=$2

# Set simulation home directory
SIM_HOME=/home/dana/Documents/SimVascular/svMultiPhysics/NoisyPipe/

# Set case directory to case name if provided, else use default naming
if [ -z "$case_name" ]; then
    case_dir=NP_$(printf "%04d" "$case")
else
    case_dir=$case_name
fi

# Visualize results
python do_vis.py --path $case_dir/ > $case_dir/vis.log
