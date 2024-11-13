epochs=(0 1 2 3 4 5 6 7 8)

for epoch in "${epochs[@]}"; do
    /workspace/wilson/Finite-groups/src/./train.py --name S5_A5x2_MLP2_128_ubias_wd2e-5_PERTURB_inst18_epoch000${epoch}00_noise0.001 --embed_dim 128 --hidden_size 128 --epochs $((20-epoch))00 --group_string "S(5);times(A(5), Z(2))" --model MLP2 --instances 30 --unembed_bias --weight_decay 2e-5 --load_weights /workspace/wilson/Finite-groups/models/2024-11-12_20-54-10_S5_A5x2_MLP2_128_ubias_wd2e-5/perturb_ckpts/inst18_epoch000${epoch}00_noise0.001.pt
done