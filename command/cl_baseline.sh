for seed in 42
do
python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/cl_baseline.py -dataset 'cub' -batch_size 128 -seed $seed -epoch 20 -num_stages 6 -concept_lambda 0.5
python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/cl_baseline.py -dataset 'awa' -batch_size 64 -seed $seed -epoch 20 -num_stages 6 -concept_lambda 0.5
done


# nohup python /hpc2hdd/home/songninglai/HZY/HZY-TimeLLM/zhanka/zhanka.py &
# nohup python /hpc2hdd/home/songninglai/HZY/HZY-TimeLLM/zhanka/zhanka.py &
# nohup python /hpc2hdd/home/songninglai/HZY/HZY-TimeLLM/zhanka/zhanka.py &
# nohup python /hpc2hdd/home/songninglai/HZY/HZY-TimeLLM/zhanka/zhanka.py &
