python -m torch.distributed.launch --nproc_per_node 2 train_volume_renderer.py --batch 12 --chunk 6 --expname afhq_sdf_vol_renderer --dataset_path ./datasets/AFHQ/train --azim 0.15 --wandb
