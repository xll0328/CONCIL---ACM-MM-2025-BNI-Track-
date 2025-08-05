## baseline, each with 5 random seed
# seed : 42 3407 114514 6666 8888


for seed in 42
do
python src/experiments/baseline.py -dataset 'cub' -batch_size 128 -seed $seed
python src/experiments/baseline.py -dataset 'awa' -batch_size 128 -seed $seed
python src/experiments/baseline.py -dataset 'cebab' -batch_size 8 -e 5 -lr 1e-5 -seed $seed
python src/experiments/baseline.py -dataset 'imdb' -batch_size 8 -e 5 -lr 1e-5 -seed $seed
done

# for seed in 42 3407 114514 6666 8888
# do
# python src/experiments/baseline.py -dataset 'cebab' -batch_size 8 -e 5 -lr 1e-5 -seed $seed
# done

# for seed in 42 3407 114514 6666 8888
# do
# python src/experiments/baseline.py -dataset 'imdb' -batch_size 8 -e 5 -lr 1e-5 -seed $seed
# done