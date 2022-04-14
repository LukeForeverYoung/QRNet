dataset=referit
python -m torch.distributed.launch --nproc_per_node=1 --master_port 23235 --use_env eval.py --data_root ./ln_data/ \
 --batch_size 8 \
 --output_dir ./outputs/$dataset \
 --dataset $dataset --max_query_len 20 \
 --eval_set test --eval_model ./outputs/$dataset/best_checkpoint.pth
