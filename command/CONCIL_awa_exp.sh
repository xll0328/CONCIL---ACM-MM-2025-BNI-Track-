


# for gg1 in 500
# do
#     for gg2 in 1
#     do
#         for bf in 25000
#         do
#             for num_stages in 2 3 4 5 6 7 8 9 10
#             do 
#             python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/CONCIL_1114.py -dataset 'awa' -gg1 $gg1 -gg2 $gg2 -saved_dir 'CONCIL_exp_cub' -buffer_size $bf -num_stages $num_stages
#             done
#         done
#     done
# done


# python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/CONCIL_1114.py -dataset 'awa' -gg1 500 -gg2 1 -saved_dir 'CONCIL_exp_awa_test' -buffer_size 5000 -num_stages 3

for gg1 in 500 
do
    for gg2 in 1
    do
        for bf in 25000
        do
            for num_stages in 2 3 4 5 6 7 8 9 10
            do 
            python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/CONCIL_1114.py -dataset 'awa' -gg1 $gg1 -gg2 $gg2 -saved_dir 'CONCIL_exp_awa_cvpr2025' -buffer_size $bf -num_stages $num_stages
            done
        done
    done
done