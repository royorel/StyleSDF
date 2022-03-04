python -m torch.distributed.launch --nproc_per_node 4 train_full_pipeline.py --batch 8 --chunk 2 --expname ffhq1024x1024 --size 1024 --wandb
