
#!/usr/bin/env bash
# Run from repo root: bash command/CONCIL_cub_exp.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$REPO_ROOT"

for gg1 in 500
do
    for gg2 in 1
    do
        for bf in 25000
        do
            for num_stages in 2 3 4 5 6 7 8 9 10
            do
                python src/experiments/CONCIL_1114.py -dataset cub -gg1 $gg1 -gg2 $gg2 -saved_dir 'CONCIL_exp_cub' -buffer_size $bf -num_stages $num_stages
            done
        done
    done
done



