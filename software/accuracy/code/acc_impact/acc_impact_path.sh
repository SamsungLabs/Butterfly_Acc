Num_Att_layer=(0 1 2 3 4)
total_iter=${#Num_Att_layer[@]}
a=0

for att_layer in "${Num_Att_layer[@]}"
do
    echo 'Running' $a '/' $total_iter 
    output=$(python3 run_tasks.py --model fft --task pathfinder32-curv_contour_length_14  --is_butterfly --hidden_dim_ratio 1.0 --transformer_dim 64 --num_layers 4 --fabnet_att_layer $att_layer --dropout_prob 0.1 2>&1 | tee -a acc_explore_path.log) 
    a=$((a + 1))
done

echo 'Explore accuracy impact on LRA-path is finished!'
