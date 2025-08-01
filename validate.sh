# # export NCCL_DEBUG=INFO
# # export NCCL_DEBUG_SUBSYS=ALL
# # export NCCL_ASYNC_ERROR_HANDLING=1
# # export NCCL_P2P_LEVEL=NVL
# # export NCCL_IB_DISABLE=1
# # export NCCL_SOCKET_IFNAME=^lo,docker0
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 validate.py --base_dir /media/NAS/DATASET/deepspeak/ --batch_size 6 --log_dir /media/NAS/USERS/shahid/MultimodalAudioVisualModerator/training_logs/aaai/multimodal/images/depth_6_heads_4_fuse_dim_16/validation --load_from /media/NAS/USERS/shahid/MultimodalAudioVisualModerator/training_logs/aaai/multimodal/images/depth_6_heads_4_fuse_dim_16/rawnet/20250604-204837
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 train.py --base_dir /media/NAS/DATASET/deepspeak/ --batch_size 3 --log_dir /media/NAS/USERS/shahid/MultimodalAudioVisualModerator/training_logs/aaai/trained_random_intra_masking_kl_divergence_self_distil_alpha_08_d_1024_gpu_4 #--load_from /media/NAS/USERS/shahid/MultimodalAudioVisualModerator/training_logs/pretrain_gpu4/rawnet/20250519-114617