
# ### multi-class classification
SEEDS=(40057 57786 60658)

# Loop through each seed in the list
for SEED in "${SEEDS[@]}"
do
	echo "################################ Running experiments with seed: $SEED ################################"

	######################################## AdaptiveViT ########################################
	
	## rho strategy: per_class
	CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py --save-name hybrid_resatt_x224 \
			--n-epochs 100 --enet-type efficientnet_b0 \
			--data-dir /media/linux-data/Workspace/Datasets/Derm7pt/ \
			--image-size 224 --batch-size 16 \
			--model-dir ./checkpoints/adavit-multi-class/Derm7pt/hybrid_resatt_x224/weights/per_class \
			--log-dir ./checkpoints/adavit-multi-class/Derm7pt/hybrid_resatt_x224/logs/per_class \
			--hybrid --hybrid-type hipervit --seed $SEED \
			--pretrained --dataset Derm7pt --out-dim 5 --rho-strategy per_class  ;

	CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python predict.py --kernel-type hybrid_resatt_x224 \
			--enet-type efficientnet_b0  \
			--data-dir /media/linux-data/Workspace/Datasets/Derm7pt/ \
			--model-dir ./checkpoints/adavit-multi-class/Derm7pt/hybrid_resatt_x224/weights/per_class \
			--sub-dir ./checkpoints/adavit-multi-class/Derm7pt/hybrid_resatt_x224/per_class \
			--image-size 224 --batch-size 16 --seed $SEED \
			--hybrid --hybrid-type hipervit \
			--eval best --out-dim 5 --dataset Derm7pt --rho-strategy per_class  ;

		
	# # ## rho strategy: per_class avg
	# # CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py --save-name hybrid_resatt_x224 \
	# # 		--n-epochs 100 --enet-type efficientnet_b0 \
	# # 		--data-dir /media/linux-data/Workspace/Datasets/Derm7pt/ \
	# # 		--image-size 224 --batch-size 16 --seed $SEED \
	# # 		--model-dir ./checkpoints/adavit-multi-class/Derm7pt/hybrid_resatt_x224/weights/per_class_avg \
	# # 		--log-dir ./checkpoints/adavit-multi-class/Derm7pt/hybrid_resatt_x224/logs/per_class_avg \
	# # 		--hybrid --hybrid-type hipervit \
	# # 		--pretrained --dataset Derm7pt --out-dim 5 --rho-strategy per_class_avg ;

	# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python predict.py --kernel-type hybrid_resatt_x224 \
	# 		--enet-type efficientnet_b0  \
	# 		--data-dir /media/linux-data/Workspace/Datasets/Derm7pt/ \
	# 		--model-dir ./checkpoints/adavit-multi-class/Derm7pt/hybrid_resatt_x224/weights/per_class_avg \
	# 		--sub-dir ./checkpoints/adavit-multi-class/Derm7pt/hybrid_resatt_x224/per_class_avg \
	# 		--image-size 224 --batch-size 16 --seed $SEED \
	# 		--hybrid --hybrid-type hipervit \
	# 		--eval best --out-dim 5 --dataset Derm7pt --rho-strategy per_class_avg ;

	# # rho strategy: per_class
	# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py --save-name hybrid_resatt_x224 \
	# 		--n-epochs 100 --enet-type efficientnet_b0 \
	# 		--data-dir /media/linux-data/Workspace/Datasets/isic_2017/ \
	# 		--image-size 224 --batch-size 64 \
	# 		--model-dir ./checkpoints/adavit-multi-class/ISIC2017/hybrid_resatt_x224/weights/per_class \
	# 		--log-dir ./checkpoints/adavit-multi-class/ISIC2017/hybrid_resatt_x224/logs/per_class \
	# 		--hybrid --hybrid-type hipervit --seed $SEED \
	# 		--pretrained --dataset ISIC2017 --out-dim 3 --rho-strategy per_class  ;

	CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python predict.py --kernel-type hybrid_resatt_x224 \
			--enet-type efficientnet_b0  \
			--data-dir /media/linux-data/Workspace/Datasets/isic_2017/ \
			--model-dir ./checkpoints/adavit-multi-class/ISIC2017/hybrid_resatt_x224/weights/per_class \
			--sub-dir ./checkpoints/adavit-multi-class/ISIC2017/hybrid_resatt_x224/per_class \
			--image-size 224 --batch-size 64 --seed $SEED \
			--hybrid --hybrid-type hipervit \
			--eval best --out-dim 3 --dataset ISIC2017 --rho-strategy per_class  ;

	# # rho strategy: per_class avg
	# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python train.py --save-name hybrid_resatt_x224 \
	# 		--n-epochs 100 --enet-type efficientnet_b0 \
	# 		--data-dir /media/linux-data/Workspace/Datasets/isic_2017/ \
	# 		--image-size 224 --batch-size 64 --seed $SEED \
	# 		--model-dir ./checkpoints/adavit-multi-class/ISIC2017/hybrid_resatt_x224/weights/per_class_avg \
	# 		--log-dir ./checkpoints/adavit-multi-class/ISIC2017/hybrid_resatt_x224/logs/per_class_avg \
	# 		--hybrid --hybrid-type hipervit \
	# 		--pretrained --dataset ISIC2017 --out-dim 3 --rho-strategy per_class_avg ;

	CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python predict.py --kernel-type hybrid_resatt_x224 \
			--enet-type efficientnet_b0  \
			--data-dir /media/linux-data/Workspace/Datasets/isic_2017/ \
			--model-dir ./checkpoints/adavit-multi-class/ISIC2017/hybrid_resatt_x224/weights/per_class_avg \
			--sub-dir ./checkpoints/adavit-multi-class/ISIC2017/hybrid_resatt_x224/per_class_avg \
			--image-size 224 --batch-size 64 --seed $SEED \
			--hybrid --hybrid-type hipervit \
			--eval best --out-dim 3 --dataset ISIC2017 --rho-strategy per_class_avg ;

done



CUDA_VISIBLE_DEVICES=0 python train.py \
		--save-name adaptivevit_Derm7pt \
		--data-dir /media/linux-data/Workspace/Datasets/Derm7pt/ \
		--dataset Derm7pt \
		--image-size 224 \
		--enet-type efficientnet_b0 \
		--out-dim 2 \
		--batch-size 32 \
		--n-epochs 30 \
		--seed 0 \
		--rho-strategy per_class_avg \
		--config ./config.yaml \
		--model-dir ./checkpoints/weights/ \
		--log-dir ./checkpoints/logs/ 

CUDA_VISIBLE_DEVICES=0 python predict.py  \
    --kernel-type adaptivevit_Derm7pt \
    --data-dir /media/linux-data/Workspace/Datasets/Derm7pt/ \
    --dataset Derm7pt \
    --image-size 224 \
    --enet-type efficientnet_b0 \
    --out-dim 2 \
    --seed 0 \
    --rho-strategy per_class_avg \
    --config ./config.yaml \
    --model-dir ./checkpoints/weights/ \
    --sub-dir ./checkpoints/subs




CUDA_VISIBLE_DEVICES=0 python train.py \
		--save-name adaptivevit_Derm7pt \
		--data-dir /media/linux-data/Workspace/Datasets/Derm7pt/ \
		--dataset Derm7pt \
		--image-size 224 \
		--enet-type efficientnet_b0 \
		--out-dim 5 \
		--batch-size 32 \
		--n-epochs 30 \
		--seed 0 \
		--rho-strategy per_class_avg \
		--config ./config.yaml \
		--model-dir ./checkpoints/weights/ \
		--log-dir ./checkpoints/logs/ 

CUDA_VISIBLE_DEVICES=0 python predict.py  \
    --kernel-type adaptivevit_Derm7pt \
    --data-dir /media/linux-data/Workspace/Datasets/Derm7pt/ \
    --dataset Derm7pt \
    --image-size 224 \
    --enet-type efficientnet_b0 \
    --out-dim 5 \
    --seed 0 \
    --rho-strategy per_class_avg \
    --config ./config.yaml \
    --model-dir ./checkpoints/weights/ \
    --sub-dir ./checkpoints/subs
