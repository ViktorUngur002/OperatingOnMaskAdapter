python train_net_maskadapter.py --num-gpus 1 --config-file configs/ground-truth-warmup/mask-adapter/mask_adapter_maft_convnext_base_cocostuff_eval_ade20k.yaml MODEL.WEIGHTS maft_weights/maftp_l.pth


python tools/weight_fuse.py --model_first_phase_path training/first-phase/maft_b_adapter/model_final.pth  --model_sem_seg_path maft_weights/maftp_b.pth --output_path weights_w_adapter/maftp_b_withadapter.pth


python train_net_maftp.py --num-gpus 1 --config-file configs/mixed-mask-training/maftp/semantic/train_semantic_base_eval_a150.yaml MODEL.WEIGHTS weights_w_adapter/maftp_l_withadapter.pth


python train_net_maftp.py --config-file configs/mixed-mask-training/maftp/semantic/train_semantic_base_eval_a150.yaml --eval-only MODEL.WEIGHTS training/maftp-base/ade20k/model_final.pth


python train_net_maftp.py --config-file configs/mixed-mask-training/maftp/semantic/eval_pc59.yaml --eval-only MODEL.WEIGHTS training/maftp-base/ade20k/model_final.pth

