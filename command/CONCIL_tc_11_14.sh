


for gg1 in 50 100 500
do
    for gg2 in 1e-2 1e-1 1 10 50 100
    do
        for bf in 25000
        do
            for num_stages in 3
            do 
            python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/CONCIL_1114.py -gg1 $gg1 -gg2 $gg2 -saved_dir 'CONCIL_tiaocan_1114' -buffer_size $bf -num_stages $num_stages
            done
        done
    done
done



