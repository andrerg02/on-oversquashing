#!/bin/bash

mkdir -p synthetic_raw_results/ring-gin
mkdir -p synthetic_raw_results/ring-sage
mkdir -p synthetic_raw_results/ring-gcn
mkdir -p synthetic_raw_results/ring-gat
mkdir -p synthetic_raw_results/ring-sheaf

for i in {2..30..2} 
    do
    for j in {1..3}
    do
        L=$((i/2))
        # GIN RING
        # if [ ! -f synthetic_raw_results/ring-gin/size-$i-seed-$j ]
        # then 
        #     python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim 64 --model gin --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/ring-gin/size-$i-seed-$j
        # fi

        # # SAGE RING
        # if [ ! -f synthetic_raw_results/ring-sage/size-$i-seed-$j ]
        # then 
        #     python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim 64 --model sage --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/ring-sage/size-$i-seed-$j
        # fi

        # # GCN RING
        # if [ ! -f synthetic_raw_results/ring-gcn/size-$i-seed-$j ]
        # then 
        #     python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim 64 --model gcn --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/ring-gcn/size-$i-seed-$j
        # fi

        # # GAT RING
        # if [ ! -f synthetic_raw_results/ring-gat/size-$i-seed-$j ]
        # then 
        #     python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim 64 --model gat --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/ring-gat/size-$i-seed-$j
        # fi

        # SHEAF RING
        if [ ! -f synthetic_raw_results/ring-sheaf/size-$i-seed-$j ]
        then 
            python exp/run.py --dataset RING --bs 128 --epochs 100 --hidden_dim 64 --model flat_bundle \
            --mpnn_layers $L --synthetic_size $i --add_crosses 0 --seed $j > synthetic_raw_results/ring-sheaf/size-$i-seed-$j
            #--sheaf_act tanh \
            # --stalk_dimension 2 --left_weights True --right_weights True --use_eps True \
            # --dropout 0.0 --use_bias False --orth householder --linear_emb False --gnn_type SAGE \
            # --gnn_layers 1 --gnn_hidden 32 --gnn_residual False --pe_size 0 \
            # --norm layer
        fi
    done
done