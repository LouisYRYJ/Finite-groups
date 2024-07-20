#!/bin/bash

# Arrays of hyperparameters
lr_values=(0.001 0.003 0.01)
beta2_values=(0.98 0.999)
weight_decay_values=(0.002 0.01)
embed_hidden_pairs=("16 32" "128 128" "1024 1024")

# run_count=0
# start_from=16

for lr in "${lr_values[@]}"; do
    for beta2 in "${beta2_values[@]}"; do
        for weight_decay in "${weight_decay_values[@]}"; do
            for pair in "${embed_hidden_pairs[@]}"; do
                # ((run_count++))
                # if [ $run_count -lt $start_from ]; then
                #     continue
                # fi
                read embed_dim hidden_size <<< "$pair"
                echo python train.py --name "xfam5_lr"$lr"_beta2"$beta2"_wd"$weight_decay"_size"$embed_dim"_"$hidden_size --group_string "XFam(5)" --instances 100 --epochs 10000 --lr "$lr" --beta2 "$beta2" --weight_decay "$weight_decay" --embed_dim "$embed_dim" --hidden_size "$hidden_size" --wandb 
                python train.py --name "xfam5_lr"$lr"_beta2"$beta2"_wd"$weight_decay"_size"$embed_dim"_"$hidden_size --group_string "XFam(5)" --instances 100 --epochs 10000 --lr "$lr" --beta2 "$beta2" --weight_decay "$weight_decay" --embed_dim "$embed_dim" --hidden_size "$hidden_size" --wandb 
            done
        done
    done
done