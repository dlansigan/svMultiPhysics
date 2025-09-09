start=$1
end=$2

for case in $(seq $start $end)
do
    ./run_case.sh $case
done