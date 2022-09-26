Num_Att_layer=(0 1 2 3 4 5 6)
total_iter=${#Num_Att_layer[@]}
a=0

for att_layer in "${Num_Att_layer[@]}"
do
    echo 'Running' $a '/' $total_iter 
    output=$(python3 run_tasks.py --model fft --task image  --is_butterfly --hidden_dim_ratio 1.0 --transformer_dim 128 --num_layers 6 --fabnet_att_layer $att_layer --dropout_prob 0.3 2>&1 | tee -a acc_explore_image.log) 
    a=$((a + 1))
done

echo 'Explore accuracy impact on LRA-image is finished!'
