start=$1
end=$2

for case in $(seq $start $end)
do
    echo $case
    ./vis_case.sh $case
done