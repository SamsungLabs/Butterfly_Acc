FFN_intern_ratio=(1.0 2.0 3.0)
Hidden_dim=(64 128 256 512 1024)
Num_layer=(2)
total_iter=$((${#FFN_intern_ratio[@]}*${#Num_layer[@]}*${#Hidden_dim[@]}))
a=0
for ratio in "${FFN_intern_ratio[@]}" #$(seq 1 ${#FFN_intern_ratio[@]})
do
    for dim in "${Hidden_dim[@]}"
    do
        for layer in "${Num_layer[@]}"
        do
            echo 'Running' $a '/' $total_iter 
	    output=$(python3 run_tasks.py --model fft --task text  --is_butterfly --hidden_dim_ratio $ratio --transformer_dim $dim --num_layers $layer --fabnet_att_layer 1 --dropout_prob 0.1 2>&1 | tee -a co_design1_text.log) 
	    a=$((a + 1))
        done
    done
done
echo 'Co-design on LRA-text is finished! Thanks!'
