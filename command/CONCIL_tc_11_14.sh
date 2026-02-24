#!/usr/bin/env bash
# Hyperparameter sweep. Run from repo root: bash command/CONCIL_tc_11_14.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO_ROOT"

for gg1 in 50 100 500
do
    for gg2 in 1e-2 1e-1 1 10 50 100
    do
        for bf in 25000
        do
            for num_stages in 3
            do
                python src/experiments/CONCIL_1114.py -dataset cub -gg1 $gg1 -gg2 $gg2 -saved_dir 'CONCIL_tiaocan_1114' -buffer_size $bf -num_stages $num_stages
            done
        done
    done
done



