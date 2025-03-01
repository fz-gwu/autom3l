
- Pytorch version is 1.11.0+cu113.

- Model will be saved to "/home/ubuntu/autom3l/autom3l_code/output/petfinder_k1_llm_20250301_184150".

- Validation metric is "accuracy".

- To track the learning progress, you can open terminal and launch Tensorboard:
    ```shell
    # Assume you have installed tensorboard
    tensorboard --logdir /home/ubuntu/autom3l/autom3l_code/output/petfinder_k1_llm_20250301_184150
    ```

Enjoy your coffee, and let AutoMM do the job     

----------------------------------
INFO:autom3l.multimodal.utils.model:selected models: ['hf_text-electra', 'numerical_transformer', 'categorical_transformer', 'timm_image-swin_transformer', 'fusion_mlp']
INFO:autom3l.multimodal.models.utils:Loading pretrained weights from Hugging Face: google/electra-base-discriminator
INFO:timm.models._builder:Loading pretrained weights from Hugging Face hub (timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k)
INFO:timm.models._hub:[timm/swin_base_patch4_window7_224.ms_in22k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
INFO:timm.models._builder:Missing keys (head.fc.weight, head.fc.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
INFO:autom3l.multimodal.predictor_llm:data_processors_count: {'text': 1, 'numerical': 1, 'categorical': 1, 'image': 1, 'label': 5}
INFO:autom3l.multimodal.predictor_llm:1 GPUs are detected, and 1 GPUs will be used.
   - GPU 0 name: Tesla T4
   - GPU 0 memory: 15.53GB/15.64GB (Free/Total)
CUDA version is 11.3.

Using 16bit None Automatic Mixed Precision (AMP)
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name              | Type               | Params
---------------------------------------------------------
0 | model             | Fusion             | 199 M 
1 | validation_metric | MulticlassAccuracy | 0     
2 | loss_func         | CrossEntropyLoss   | 0     
---------------------------------------------------------
199 M     Trainable params
0         Non-trainable params
199 M     Total params
399.944   Total estimated model params size (MB)
Epoch 0:  50%|██████████████████████████████████████████████████▍                                                  | 1007/2015 [04:03<04:03,  4.14it/s, loss=nan, v_num=Epoch 0, global step 51: 'val_accuracy' reached 0.02055 (best 0.02055), saving model to '/home/ubuntu/autom3l/autom3l_code/output/petfinder_k1_llm_20250301_184150/epoch=0-step=51.ckpt' as top 3
Epoch 0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████▉| 2014/2015 [08:13<00:00,  4.08it/s, loss=nan, v_num=Epoch 0, global step 103: 'val_accuracy' reached 0.02055 (best 0.02055), saving model to '/home/ubuntu/autom3l/autom3l_code/output/petfinder_k1_llm_20250301_184150/epoch=0-step=103.ckpt' as top 3
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 2015/2015 [08:24<00:00,  3.99it/s, loss=nan, v_num=]`Trainer.fit` stopped: `max_epochs=1` reached.
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 2015/2015 [08:24<00:00,  3.99it/s, loss=nan, v_num=]
INFO:autom3l.multimodal.predictor_llm:Start to fuse 2 checkpoints via the greedy soup algorithm.
Predicting DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 46/46 [00:21<00:00,  2.13it/s]
Predicting DataLoader 0: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 46/46 [00:21<00:00,  2.12it/s]
INFO:autom3l.multimodal.predictor_llm:AutoMM has created your model    

- To load the model, use the code below:
    ```python
    from autom3l.multimodal import MultiModalPredictor
    predictor = MultiModalPredictor.load("/home/ubuntu/autom3l/autom3l_code/output/petfinder_k1_llm_20250301_184150")
    ```

- You can open terminal and launch Tensorboard to visualize the training log:
    ```shell
    # Assume you have installed tensorboard
    tensorboard --logdir /home/ubuntu/autom3l/autom3l_code/output/petfinder_k1_llm_20250301_184150
