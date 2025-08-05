


for gg in 1e-1 1e-2 1e-3 1e-4 1e-0
do
    for bf in 16000 24000 32000 40000 48000 56000 64000 72000 80000 8000 100000 200000 300000 400000
    do
        for num_stages in 2 3 4 5 6 7 8 9 10
        do 
        python /hpc2hdd/home/songninglai/CBM_CL/src/experiments/CONCIL_1111.py -gg $gg -saved_dir 'results_CONCIL' -buffer_size $bf -num_stages $num_stages
        done
    done
done


conda activate tbw

nohup python /hpc2hdd/home/songninglai/ParamAttack/gpu_monitor.py 50 > zk.log &

