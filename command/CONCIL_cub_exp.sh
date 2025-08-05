


for gg1 in 500
do
    for gg2 in 1
    do
        for bf in 25000
        do
            for num_stages in 2 3 4 5 6 7 8 9 10
            do 
            python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/CONCIL_1114.py -gg1 $gg1 -gg2 $gg2 -saved_dir 'CONCIL_exp_cub' -buffer_size $bf -num_stages $num_stages
            done
        done
    done
done



