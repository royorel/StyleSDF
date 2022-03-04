python -m torch.distributed.launch --nproc_per_node 4 new_train.py --batch 8 --chunk 4 --azim 0.15 --r1 50.0 --expname afhq512x512 --dataset_path ./datasets/AFHQ/train/ --size 512 --wandb
