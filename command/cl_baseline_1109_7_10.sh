# for seed in 42
# do
# python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/cl_baseline.py -dataset 'cub' -batch_size 128 -seed $seed -epoch 30 -num_stages 6 -concept_lambda 0.5
# python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/cl_baseline.py -dataset 'awa' -batch_size 64 -seed $seed -epoch 30 -num_stages 6 -concept_lambda 0.5
# done


# for epoch in 20 30
# do 
#     for num_stages in 3 4 5 6 7 8 9 10
#     do
#     python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/cl_baseline.py -dataset 'cub' -batch_size 128 -epoch $epoch -num_stages $num_stages -concept_lambda 0.5
#     python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/cl_baseline.py -dataset 'awa' -batch_size 64 -epoch $epoch -num_stages $num_stages -concept_lambda 0.5
#     done
# done

# for epoch in 20
# do 
#     for num_stages in 5 6 7 8 9 10
#     do
#     python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/cl_baseline.py -dataset 'cub' -batch_size 128 -epoch $epoch -num_stages $num_stages -concept_lambda 0.5
#     python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/cl_baseline.py -dataset 'awa' -batch_size 128 -epoch $epoch -num_stages $num_stages -concept_lambda 0.5
#     done
# done


# for epoch in 30
# do 
#     for num_stages in 2 3 4 5 6
#     do
#     python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/cl_baseline.py -dataset 'cub' -batch_size 128 -epoch $epoch -num_stages $num_stages -concept_lambda 0.5
#     python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/cl_baseline.py -dataset 'awa' -batch_size 128 -epoch $epoch -num_stages $num_stages -concept_lambda 0.5
#     done
# done

for epoch in 30
do 
    for num_stages in 7 8 9 10
    do
    python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/cl_baseline.py -dataset 'cub' -batch_size 128 -epoch $epoch -num_stages $num_stages -concept_lambda 0.5
    python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/cl_baseline.py -dataset 'awa' -batch_size 128 -epoch $epoch -num_stages $num_stages -concept_lambda 0.5
    done
done


# nohup python /hpc2hdd/home/songninglai/HZY/HZY-TimeLLM/zhanka/zhanka.py &
# nohup python /hpc2hdd/home/songninglai/HZY/HZY-TimeLLM/zhanka/zhanka.py &
# nohup python /hpc2hdd/home/songninglai/HZY/HZY-TimeLLM/zhanka/zhanka.py &
# nohup python /hpc2hdd/home/songninglai/HZY/HZY-TimeLLM/zhanka/zhanka.py &
