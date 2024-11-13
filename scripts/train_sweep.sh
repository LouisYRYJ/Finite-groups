#!/bin/bash

# Arrays of hyperparameters
lr_values=(5e-4 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 1e-1)
# lr_values=(0.003)
beta2_values=(0.999)
weight_decay_values=(1e-4)
# weight_decay_values=(0 1e-7 3e-7 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2)
embed_hidden_pairs=("128 128")

# Fixed variables
project='abFam(5, 7) sweep lr_only decay=1e-4'
name='abfam5_7'
group_string='abFam(5, 7)'
epochs=5000

# run_count=0
# start_from=0

for lr in "${lr_values[@]}"; do
    for beta2 in "${beta2_values[@]}"; do
        for weight_decay in "${weight_decay_values[@]}"; do
            for pair in "${embed_hidden_pairs[@]}"; do
                # ((run_count++))
                # if [ $run_count -lt $start_from ]; then
                #     continue
                # fi
                read embed_dim hidden_size <<< "$pair"
                echo python train.py --project "$project" --name $name"_lr"$lr"_beta2"$beta2"_wd"$weight_decay"_size"$embed_dim"_"$hidden_size --group_string "$group_string" --instances 100 --epochs $epochs --lr "$lr" --beta2 "$beta2" --weight_decay "$weight_decay" --embed_dim "$embed_dim" --hidden_size "$hidden_size" --wandb 
                python train.py --project "$project" --name $name"_lr"$lr"_beta2"$beta2"_wd"$weight_decay"_size"$embed_dim"_"$hidden_size --group_string "$group_string" --instances 100 --epochs $epochs --lr "$lr" --beta2 "$beta2" --weight_decay "$weight_decay" --embed_dim "$embed_dim" --hidden_size "$hidden_size" --wandb 
            done
        done
    done
done