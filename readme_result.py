----------------
fold: 1
path: ./output/petfinder_k1_llm_20250302_234413
preset: llm
hpo_trials: 1
gpu: 0
--------------
/home/ubuntu/autom3l/autom3l_code/multimodal/src/autom3l/multimodal/presets.py:49: LangChainDeprecationWarning: Importing OpenAIEmbeddings from langchain.embeddings is deprecated. Please replace deprecated imports:

>> from langchain.embeddings import OpenAIEmbeddings

with new imports of:

>> from langchain_community.embeddings import OpenAIEmbeddings
You can use the langchain cli to **automatically** upgrade many imports. Please see documentation here <https://python.langchain.com/v0.2/docs/versions/v0_2/>
  from langchain.embeddings.openai import OpenAIEmbeddings
/home/ubuntu/autom3l/autom3l_code/multimodal/src/autom3l/multimodal/presets.py:50: LangChainDeprecationWarning: Importing FAISS from langchain.vectorstores is deprecated. Please replace deprecated imports:

>> from langchain.vectorstores import FAISS

with new imports of:

>> from langchain_community.vectorstores import FAISS
You can use the langchain cli to **automatically** upgrade many imports. Please see documentation here <https://python.langchain.com/v0.2/docs/versions/v0_2/>
  from langchain.vectorstores.faiss import FAISS
INFO:torch.distributed.nn.jit.instantiator:Created a temporary directory at /tmp/tmpv9vl4k1j
INFO:torch.distributed.nn.jit.instantiator:Writing /tmp/tmpv9vl4k1j/_remote_module_non_sriptable.py
/home/ubuntu/autom3l/autom3l_code/multimodal/src/autom3l/multimodal/predictor_llm.py:185: LangChainDeprecationWarning: Importing OpenAIEmbeddings from langchain.embeddings is deprecated. Please replace deprecated imports:

>> from langchain.embeddings import OpenAIEmbeddings

with new imports of:

>> from langchain_community.embeddings import OpenAIEmbeddings
You can use the langchain cli to **automatically** upgrade many imports. Please see documentation here <https://python.langchain.com/v0.2/docs/versions/v0_2/>
  from langchain.embeddings.openai import OpenAIEmbeddings
/home/ubuntu/autom3l/autom3l_code/multimodal/src/autom3l/multimodal/predictor_llm.py:186: LangChainDeprecationWarning: Importing FAISS from langchain.vectorstores is deprecated. Please replace deprecated imports:

>> from langchain.vectorstores import FAISS

with new imports of:

>> from langchain_community.vectorstores import FAISS
You can use the langchain cli to **automatically** upgrade many imports. Please see documentation here <https://python.langchain.com/v0.2/docs/versions/v0_2/>
  from langchain.vectorstores.faiss import FAISS
/home/ubuntu/autom3l/autom3l_code/ag_automm_tutorial
INFO:autom3l.multimodal.data.data_llm:loading data:/home/ubuntu/autom3l/autom3l_code/ag_automm_tutorial/petfinder/10-fold/train_fold_10.csv
INFO:autom3l.multimodal.utils.save:save path: /home/ubuntu/autom3l/autom3l_code/output/petfinder_k1_llm_20250302_234413
INFO:autom3l.multimodal.predictor_llm:show traindata:
              Name  Age  Breed1  Breed2  ...      PetID  PhotoAmt  AdoptionSpeed                                             Images
Type                                    ...                                                                                       
2          Nibble    3     299       0  ...  86e1089a3       1.0              2  /home/ubuntu/autom3l/autom3l_code/ag_automm_tu...
2     No Name Yet    1     265       0  ...  6296e909a       2.0              0  /home/ubuntu/autom3l/autom3l_code/ag_automm_tu...
1          Brisco    1     307       0  ...  3422e4906       7.0              3  /home/ubuntu/autom3l/autom3l_code/ag_automm_tu...

[3 rows x 24 columns]
INFO:autom3l.multimodal.predictor_llm:running MI-LLM...
/home/ubuntu/autom3l/autom3l_code/multimodal/src/autom3l/multimodal/predictor_llm.py:869: LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use RunnableSequence, e.g., `prompt | llm` instead.
  location_chain = LLMChain(llm=self.llm, prompt=prompt_template)
/home/ubuntu/autom3l/autom3l_code/multimodal/src/autom3l/multimodal/predictor_llm.py:870: LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use invoke instead.
  column_types_llm = location_chain.run(
INFO:autom3l.multimodal.predictor_llm:✨MI-LLM results✨:
 {'Name': 'text', 'Age': 'numerical', 'Breed1': 'numerical', 'Breed2': 'numerical', 'Gender': 'categorical', 'Color1': 'numerical', 'Color2': 'numerical', 'Color3': 'numerical', 'MaturitySize': 'categorical', 'FurLength': 'categorical', 'Vaccinated': 'categorical', 'Dewormed': 'categorical', 'Sterilized': 'categorical', 'Health': 'categorical', 'Quantity': 'numerical', 'Fee': 'numerical', 'State': 'numerical', 'RescuerID': 'text', 'VideoAmt': 'numerical', 'Description': 'text', 'PetID': 'text', 'PhotoAmt': 'numerical', 'AdoptionSpeed': 'categorical', 'Images': 'image_path'}
INFO:autom3l.multimodal.predictor_llm:val metric:accuracy;eval metric:accuracy
INFO:autom3l.multimodal.presets:RUN MS-LLM...
INFO:autom3l.multimodal.presets:✨USER_INSTRUCTION✨: I want a model with best accuracy.

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1872.46it/s]
/home/ubuntu/autom3l/autom3l_code/multimodal/src/autom3l/multimodal/presets.py:384: LangChainDeprecationWarning: The class `OpenAIEmbeddings` was deprecated in LangChain 0.0.9 and will be removed in 1.0. An updated version of the class exists in the langchain-openai package and should be used instead. To use it run `pip install -U langchain-openai` and import as `from langchain_openai import OpenAIEmbeddings`.
  embeddings = OpenAIEmbeddings()
INFO:faiss.loader:Loading faiss with AVX512 support.
INFO:faiss.loader:Successfully loaded faiss with AVX512 support.
INFO:faiss:Failed to load GPU Faiss: name 'GpuIndexIVFFlat' is not defined. Will not load constructor refs for GPU indexes.
INFO:autom3l.multimodal.presets:✨MODEL SELECTED✨:
 {'name': 'hf_text-electra', 'reason': 'ELECTRA has shown high accuracy on various NLP tasks, including text classification. It outperforms other models on benchmarks like GLUE and SQuAD 2.0, making it a strong candidate for achieving the best accuracy in classifying animal adoption rates based on text data.'}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 2052.01it/s]
INFO:autom3l.multimodal.presets:✨MODEL SELECTED✨:
 {'name': 'numerical_transformer', 'reason': 'FT-Transformer has demonstrated the best performance on most tabular deep learning tasks, and it is specifically designed for handling tabular data with both numerical and categorical features. Given that the dataset includes tabular features such as age, breed, name, and color, FT-Transformer is likely to provide the best accuracy for the classification task of predicting animal adoption rates.'}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 1896.59it/s]
INFO:autom3l.multimodal.presets:✨MODEL SELECTED✨:
 {'name': 'categorical_transformer', 'reason': "FT-Transformer has demonstrated the best performance on most tabular deep learning tasks, and it is specifically designed for handling tabular data with both numerical and categorical features. Given that the dataset includes text, tabular, and image data, FT-Transformer's ability to transform all features to embeddings and apply a stack of Transformer layers makes it a strong candidate for achieving the best accuracy in this classification task."}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 1204.74it/s]
INFO:autom3l.multimodal.presets:✨MODEL SELECTED✨:
 {'name': 'timm_image-swin_transformer', 'reason': 'Swin Transformer has achieved impressive results in various vision tasks, including image classification, object detection, and semantic segmentation. It has state-of-the-art accuracy on tasks such as ImageNet-1K classification, COCO object detection, and ADE20K semantic segmentation. With its hierarchical architecture and shifted windowing scheme, Swin Transformer is well-suited for handling multimodal data like images, text, and tabular features in the PetFinder.my dataset.'}
INFO:autom3l.multimodal.presets:presets:llm,use_hpo:False
INFO:autom3l.multimodal.predictor_llm:hpo spaces(before filtering):
INFO:autom3l.multimodal.predictor_llm:hyperparameters:{'model.names': ['hf_text-electra', 'numerical_transformer', 'categorical_transformer', 'timm_image-swin_transformer', 'fusion_mlp']} 
INFO:autom3l.multimodal.predictor_llm:hyperparameter_tune_kwargs:{}
INFO:autom3l.multimodal.predictor_llm:hpo spaces(after model filtering):
INFO:autom3l.multimodal.predictor_llm:hyperparameters:{'model.names': ['hf_text-electra', 'numerical_transformer', 'categorical_transformer', 'timm_image-swin_transformer', 'fusion_mlp']} 
Global seed set to 0
INFO:autom3l.multimodal.predictor_llm:AutoMM starts to create your model.  

- AutoM3L version is 0.8.1b20250228.

- Pytorch version is 1.11.0+cu113.

- Model will be saved to "/home/ubuntu/autom3l/autom3l_code/output/petfinder_k1_llm_20250302_234413".

- Validation metric is "accuracy".

- To track the learning progress, you can open terminal and launch Tensorboard:
    ```shell
    # Assume you have installed tensorboard
    tensorboard --logdir /home/ubuntu/autom3l/autom3l_code/output/petfinder_k1_llm_20250302_234413
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
399.950   Total estimated model params size (MB)
Epoch 0:  50%|████████████████████████████████████████████▍                                            | 1007/2015 [04:26<04:26,  3.78it/s, loss=1.87, v_num=Epoch 0, global step 51: 'val_accuracy' reached 0.35548 (best 0.35548), saving model to '/home/ubuntu/autom3l/autom3l_code/output/petfinder_k1_llm_20250302_234413/epoch=0-step=51.ckpt' as top 3
Epoch 0: 100%|████████████████████████████████████████████████████████████████████████████████████████▉| 2014/2015 [09:21<00:00,  3.59it/s, loss=2.02, v_num=Epoch 0, global step 103: 'val_accuracy' reached 0.36986 (best 0.36986), saving model to '/home/ubuntu/autom3l/autom3l_code/output/petfinder_k1_llm_20250302_234413/epoch=0-step=103.ckpt' as top 3
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 2015/2015 [10:01<00:00,  3.35it/s, loss=2.01, v_num=]`Trainer.fit` stopped: `max_epochs=1` reached.
Epoch 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 2015/2015 [10:01<00:00,  3.35it/s, loss=2.01, v_num=]
INFO:autom3l.multimodal.predictor_llm:Start to fuse 2 checkpoints via the greedy soup algorithm.
Predicting DataLoader 0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 46/46 [00:22<00:00,  2.05it/s]
Predicting DataLoader 0: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 46/46 [00:22<00:00,  2.04it/s]
INFO:autom3l.multimodal.predictor_llm:AutoMM has created your model    

- To load the model, use the code below:
    ```python
    from autom3l.multimodal import MultiModalPredictor
    predictor = MultiModalPredictor.load("/home/ubuntu/autom3l/autom3l_code/output/petfinder_k1_llm_20250302_234413")
    ```

- You can open terminal and launch Tensorboard to visualize the training log:
    ```shell
    # Assume you have installed tensorboard
    tensorboard --logdir /home/ubuntu/autom3l/autom3l_code/output/petfinder_k1_llm_20250302_234413
    ```
