multiclass
num_gpus: 1
strategy: None
df_preprocessor: MultiModalFeaturePreprocessor(column_types={'AdoptionSpeed': 'categorical',
                                            'Age': 'numerical',
                                            'Breed1': 'numerical',
                                            'Breed2': 'numerical',
                                            'Color1': 'numerical',
                                            'Color2': 'numerical',
                                            'Color3': 'numerical',
                                            'Description': 'text',
                                            'Dewormed': 'categorical',
                                            'Fee': 'numerical',
                                            'FurLength': 'categorical',
                                            'Gender': 'categorical',
                                            'Health': 'categorical',
                                            'Images': 'image_p...
                              config={'image': {'missing_value_strategy': 'zero'}, 'text': {'normalize_text': False}, 'categorical': {'minimum_cat_count': 100, 'maximum_num_cat': 20, 'convert_to_text': False}, 'numerical': {'convert_to_text': False, 'scaler_with_mean': True, 'scaler_with_std': True}, 'document': {'missing_value_strategy': 'zero'}, 'label': {'numerical_label_preprocessing': 'standardscaler'}, 'pos_label': None, 'mixup': {'turn_on': False, 'mixup_alpha': 0.8, 'cutmix_alpha': 1.0, 'cutmix_minmax': None, 'prob': 1.0, 'switch_prob': 0.5, 'mode': 'batch', 'turn_off_epoch': 5, 'label_smoothing': 0.1}, 'templates': {'turn_on': False, 'num_templates': 30, 'template_length': 2048, 'preset_templates': ['super_glue', 'rte'], 'custom_templates': None}, 'video': {'missing_value_strategy': 'zero', 'requires_column_info': False}},
                              label_column='AdoptionSpeed',
                              label_generator=<autom3l.multimodal.data.label_encoder.CustomLabelEncoder object at 0x7ff296362f40>)
data_processors: {'text': [<autom3l.multimodal.data.process_text.TextProcessor_llm object at 0x7ff296362d90>], 'numerical': [<autom3l.multimodal.data.process_numerical.NumericalProcessor_llm object at 0x7ff2963d7730>], 'categorical': [<autom3l.multimodal.data.process_categorical.CategoricalProcessor_llm object at 0x7ff2963d7790>], 'image': [<autom3l.multimodal.data.process_image.ImageProcessor_llm object at 0x7ff2963d77f0>], 'label': [<autom3l.multimodal.data.process_label._LabelProcessor_llm object at 0x7ff2963d7fd0>, <autom3l.multimodal.data.process_label._LabelProcessor_llm object at 0x7ff2963e6070>, <autom3l.multimodal.data.process_label._LabelProcessor_llm object at 0x7ff2963e60d0>, <autom3l.multimodal.data.process_label._LabelProcessor_llm object at 0x7ff2963e6130>, <autom3l.multimodal.data.process_label._LabelProcessor_llm object at 0x7ff2963e6190>]}
predict_dm: <autom3l.multimodal.data.datamodule.BaseDataModule object at 0x7ff3a977b1c0>
self._model: MultimodalFusionMLP(
  (model): ModuleList(
    (0): CategoricalTransformer(
      (categorical_feature_tokenizer): CategoricalFeatureTokenizer(
        (embeddings): Embedding(27, 192)
      )
      (cls_token): Identity()
      (transformer): FT_Transformer(
        (blocks): ModuleList(
          (0): ModuleDict(
            (attention): MultiheadAttention(
              (W_q): Linear(in_features=192, out_features=192, bias=True)
              (W_k): Linear(in_features=192, out_features=192, bias=True)
              (W_v): Linear(in_features=192, out_features=192, bias=True)
              (W_out): Linear(in_features=192, out_features=192, bias=True)
              (dropout): Dropout(p=0.2, inplace=False)
            )
            (ffn): FFN(
              (linear_first): Linear(in_features=192, out_features=384, bias=True)
              (activation): ReGLU()
              (dropout): Dropout(p=0.1, inplace=False)
              (linear_second): Linear(in_features=192, out_features=192, bias=True)
            )
            (attention_residual_dropout): Dropout(p=0.0, inplace=False)
            (ffn_residual_dropout): Dropout(p=0.0, inplace=False)
            (output): Identity()
            (ffn_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (1): ModuleDict(
            (attention): MultiheadAttention(
              (W_q): Linear(in_features=192, out_features=192, bias=True)
              (W_k): Linear(in_features=192, out_features=192, bias=True)
              (W_v): Linear(in_features=192, out_features=192, bias=True)
              (W_out): Linear(in_features=192, out_features=192, bias=True)
              (dropout): Dropout(p=0.2, inplace=False)
            )
            (ffn): FFN(
              (linear_first): Linear(in_features=192, out_features=384, bias=True)
              (activation): ReGLU()
              (dropout): Dropout(p=0.1, inplace=False)
              (linear_second): Linear(in_features=192, out_features=192, bias=True)
            )
            (attention_residual_dropout): Dropout(p=0.0, inplace=False)
            (ffn_residual_dropout): Dropout(p=0.0, inplace=False)
            (output): Identity()
            (attention_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (ffn_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (2): ModuleDict(
            (attention): MultiheadAttention(
              (W_q): Linear(in_features=192, out_features=192, bias=True)
              (W_k): Linear(in_features=192, out_features=192, bias=True)
              (W_v): Linear(in_features=192, out_features=192, bias=True)
              (W_out): Linear(in_features=192, out_features=192, bias=True)
              (dropout): Dropout(p=0.2, inplace=False)
            )
            (ffn): FFN(
              (linear_first): Linear(in_features=192, out_features=384, bias=True)
              (activation): ReGLU()
              (dropout): Dropout(p=0.1, inplace=False)
              (linear_second): Linear(in_features=192, out_features=192, bias=True)
            )
            (attention_residual_dropout): Dropout(p=0.0, inplace=False)
            (ffn_residual_dropout): Dropout(p=0.0, inplace=False)
            (output): Identity()
            (attention_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (ffn_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
        )
        (head): Identity()
      )
      (head): Head(
        (normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        (activation): ReLU()
        (linear): Linear(in_features=192, out_features=5, bias=True)
      )
    )
    (1): HFAutoModelForTextPrediction(
      (model): ElectraModel(
        (embeddings): ElectraEmbeddings(
          (word_embeddings): Embedding(30522, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (token_type_embeddings): Embedding(2, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): ElectraEncoder(
          (layer): ModuleList(
            (0): ElectraLayer(
              (attention): ElectraAttention(
                (self): ElectraSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): ElectraSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): ElectraIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): ElectraOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (1): ElectraLayer(
              (attention): ElectraAttention(
                (self): ElectraSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): ElectraSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): ElectraIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): ElectraOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (2): ElectraLayer(
              (attention): ElectraAttention(
                (self): ElectraSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): ElectraSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): ElectraIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): ElectraOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (3): ElectraLayer(
              (attention): ElectraAttention(
                (self): ElectraSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): ElectraSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): ElectraIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): ElectraOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (4): ElectraLayer(
              (attention): ElectraAttention(
                (self): ElectraSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): ElectraSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): ElectraIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): ElectraOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (5): ElectraLayer(
              (attention): ElectraAttention(
                (self): ElectraSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): ElectraSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): ElectraIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): ElectraOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (6): ElectraLayer(
              (attention): ElectraAttention(
                (self): ElectraSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): ElectraSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): ElectraIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): ElectraOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (7): ElectraLayer(
              (attention): ElectraAttention(
                (self): ElectraSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): ElectraSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): ElectraIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): ElectraOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (8): ElectraLayer(
              (attention): ElectraAttention(
                (self): ElectraSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): ElectraSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): ElectraIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): ElectraOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (9): ElectraLayer(
              (attention): ElectraAttention(
                (self): ElectraSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): ElectraSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): ElectraIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): ElectraOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (10): ElectraLayer(
              (attention): ElectraAttention(
                (self): ElectraSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): ElectraSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): ElectraIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): ElectraOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (11): ElectraLayer(
              (attention): ElectraAttention(
                (self): ElectraSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): ElectraSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): ElectraIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): ElectraOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
      )
      (head): Linear(in_features=768, out_features=5, bias=True)
    )
    (2): NumericalTransformer(
      (numerical_feature_tokenizer): NumEmbeddings(
        (layers): Sequential(
          (0): NumericalFeatureTokenizer()
          (1): ReLU()
        )
      )
      (cls_token): Identity()
      (transformer): FT_Transformer(
        (blocks): ModuleList(
          (0): ModuleDict(
            (attention): MultiheadAttention(
              (W_q): Linear(in_features=192, out_features=192, bias=True)
              (W_k): Linear(in_features=192, out_features=192, bias=True)
              (W_v): Linear(in_features=192, out_features=192, bias=True)
              (W_out): Linear(in_features=192, out_features=192, bias=True)
              (dropout): Dropout(p=0.2, inplace=False)
            )
            (ffn): FFN(
              (linear_first): Linear(in_features=192, out_features=384, bias=True)
              (activation): ReGLU()
              (dropout): Dropout(p=0.1, inplace=False)
              (linear_second): Linear(in_features=192, out_features=192, bias=True)
            )
            (attention_residual_dropout): Dropout(p=0.0, inplace=False)
            (ffn_residual_dropout): Dropout(p=0.0, inplace=False)
            (output): Identity()
            (ffn_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (1): ModuleDict(
            (attention): MultiheadAttention(
              (W_q): Linear(in_features=192, out_features=192, bias=True)
              (W_k): Linear(in_features=192, out_features=192, bias=True)
              (W_v): Linear(in_features=192, out_features=192, bias=True)
              (W_out): Linear(in_features=192, out_features=192, bias=True)
              (dropout): Dropout(p=0.2, inplace=False)
            )
            (ffn): FFN(
              (linear_first): Linear(in_features=192, out_features=384, bias=True)
              (activation): ReGLU()
              (dropout): Dropout(p=0.1, inplace=False)
              (linear_second): Linear(in_features=192, out_features=192, bias=True)
            )
            (attention_residual_dropout): Dropout(p=0.0, inplace=False)
            (ffn_residual_dropout): Dropout(p=0.0, inplace=False)
            (output): Identity()
            (attention_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (ffn_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
          (2): ModuleDict(
            (attention): MultiheadAttention(
              (W_q): Linear(in_features=192, out_features=192, bias=True)
              (W_k): Linear(in_features=192, out_features=192, bias=True)
              (W_v): Linear(in_features=192, out_features=192, bias=True)
              (W_out): Linear(in_features=192, out_features=192, bias=True)
              (dropout): Dropout(p=0.2, inplace=False)
            )
            (ffn): FFN(
              (linear_first): Linear(in_features=192, out_features=384, bias=True)
              (activation): ReGLU()
              (dropout): Dropout(p=0.1, inplace=False)
              (linear_second): Linear(in_features=192, out_features=192, bias=True)
            )
            (attention_residual_dropout): Dropout(p=0.0, inplace=False)
            (ffn_residual_dropout): Dropout(p=0.0, inplace=False)
            (output): Identity()
            (attention_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (ffn_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          )
        )
        (head): Identity()
      )
      (head): Head(
        (normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        (activation): ReLU()
        (linear): Linear(in_features=192, out_features=5, bias=True)
      )
    )
    (3): TimmAutoModelForImagePrediction(
      (model): SwinTransformer(
        (patch_embed): PatchEmbed(
          (proj): Conv2d(3, 128, kernel_size=(4, 4), stride=(4, 4))
          (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        )
        (layers): Sequential(
          (0): SwinTransformerStage(
            (downsample): Identity()
            (blocks): Sequential(
              (0): SwinTransformerBlock(
                (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=128, out_features=384, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=128, out_features=128, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): Identity()
                (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=128, out_features=512, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=512, out_features=128, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): Identity()
              )
              (1): SwinTransformerBlock(
                (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=128, out_features=384, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=128, out_features=128, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.004)
                (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=128, out_features=512, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=512, out_features=128, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.004)
              )
            )
          )
          (1): SwinTransformerStage(
            (downsample): PatchMerging(
              (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
              (reduction): Linear(in_features=512, out_features=256, bias=False)
            )
            (blocks): Sequential(
              (0): SwinTransformerBlock(
                (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=256, out_features=768, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=256, out_features=256, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.009)
                (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=256, out_features=1024, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=1024, out_features=256, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.009)
              )
              (1): SwinTransformerBlock(
                (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=256, out_features=768, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=256, out_features=256, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.013)
                (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=256, out_features=1024, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=1024, out_features=256, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.013)
              )
            )
          )
          (2): SwinTransformerStage(
            (downsample): PatchMerging(
              (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              (reduction): Linear(in_features=1024, out_features=512, bias=False)
            )
            (blocks): Sequential(
              (0): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.017)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.017)
              )
              (1): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.022)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.022)
              )
              (2): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.026)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.026)
              )
              (3): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.030)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.030)
              )
              (4): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.035)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.035)
              )
              (5): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.039)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.039)
              )
              (6): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.043)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.043)
              )
              (7): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.048)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.048)
              )
              (8): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.052)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.052)
              )
              (9): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.057)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.057)
              )
              (10): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.061)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.061)
              )
              (11): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.065)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.065)
              )
              (12): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.070)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.070)
              )
              (13): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.074)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.074)
              )
              (14): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.078)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.078)
              )
              (15): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.083)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.083)
              )
              (16): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.087)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.087)
              )
              (17): SwinTransformerBlock(
                (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=512, out_features=1536, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=512, out_features=512, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.091)
                (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=512, out_features=2048, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=2048, out_features=512, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.091)
              )
            )
          )
          (3): SwinTransformerStage(
            (downsample): PatchMerging(
              (norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
              (reduction): Linear(in_features=2048, out_features=1024, bias=False)
            )
            (blocks): Sequential(
              (0): SwinTransformerBlock(
                (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=1024, out_features=3072, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=1024, out_features=1024, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.096)
                (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=1024, out_features=4096, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=4096, out_features=1024, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.096)
              )
              (1): SwinTransformerBlock(
                (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (attn): WindowAttention(
                  (qkv): Linear(in_features=1024, out_features=3072, bias=True)
                  (attn_drop): Dropout(p=0.0, inplace=False)
                  (proj): Linear(in_features=1024, out_features=1024, bias=True)
                  (proj_drop): Dropout(p=0.0, inplace=False)
                  (softmax): Softmax(dim=-1)
                )
                (drop_path1): DropPath(drop_prob=0.100)
                (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (mlp): Mlp(
                  (fc1): Linear(in_features=1024, out_features=4096, bias=True)
                  (act): GELU()
                  (drop1): Dropout(p=0.0, inplace=False)
                  (norm): Identity()
                  (fc2): Linear(in_features=4096, out_features=1024, bias=True)
                  (drop2): Dropout(p=0.0, inplace=False)
                )
                (drop_path2): DropPath(drop_prob=0.100)
              )
            )
          )
        )
        (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (head): ClassifierHead(
          (global_pool): SelectAdaptivePool2d(pool_type=avg, flatten=Identity())
          (drop): Dropout(p=0.0, inplace=False)
          (fc): Identity()
          (flatten): Identity()
        )
      )
      (head): Linear(in_features=1024, out_features=5, bias=True)
    )
  )
  (adapter): ModuleList(
    (0): Linear(in_features=192, out_features=1024, bias=True)
    (1): Linear(in_features=768, out_features=1024, bias=True)
    (2): Linear(in_features=192, out_features=1024, bias=True)
    (3): Linear(in_features=1024, out_features=1024, bias=True)
  )
  (fusion_mlp): Sequential(
    (0): MLP(
      (layers): Sequential(
        (0): Unit(
          (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
          (fc): Linear(in_features=4096, out_features=128, bias=True)
          (act_fn): LeakyReLU(negative_slope=0.01)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (head): Linear(in_features=128, out_features=5, bias=True)
)
task: LitModule(
  (model): MultimodalFusionMLP(
    (model): ModuleList(
      (0): CategoricalTransformer(
        (categorical_feature_tokenizer): CategoricalFeatureTokenizer(
          (embeddings): Embedding(27, 192)
        )
        (cls_token): Identity()
        (transformer): FT_Transformer(
          (blocks): ModuleList(
            (0): ModuleDict(
              (attention): MultiheadAttention(
                (W_q): Linear(in_features=192, out_features=192, bias=True)
                (W_k): Linear(in_features=192, out_features=192, bias=True)
                (W_v): Linear(in_features=192, out_features=192, bias=True)
                (W_out): Linear(in_features=192, out_features=192, bias=True)
                (dropout): Dropout(p=0.2, inplace=False)
              )
              (ffn): FFN(
                (linear_first): Linear(in_features=192, out_features=384, bias=True)
                (activation): ReGLU()
                (dropout): Dropout(p=0.1, inplace=False)
                (linear_second): Linear(in_features=192, out_features=192, bias=True)
              )
              (attention_residual_dropout): Dropout(p=0.0, inplace=False)
              (ffn_residual_dropout): Dropout(p=0.0, inplace=False)
              (output): Identity()
              (ffn_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            )
            (1): ModuleDict(
              (attention): MultiheadAttention(
                (W_q): Linear(in_features=192, out_features=192, bias=True)
                (W_k): Linear(in_features=192, out_features=192, bias=True)
                (W_v): Linear(in_features=192, out_features=192, bias=True)
                (W_out): Linear(in_features=192, out_features=192, bias=True)
                (dropout): Dropout(p=0.2, inplace=False)
              )
              (ffn): FFN(
                (linear_first): Linear(in_features=192, out_features=384, bias=True)
                (activation): ReGLU()
                (dropout): Dropout(p=0.1, inplace=False)
                (linear_second): Linear(in_features=192, out_features=192, bias=True)
              )
              (attention_residual_dropout): Dropout(p=0.0, inplace=False)
              (ffn_residual_dropout): Dropout(p=0.0, inplace=False)
              (output): Identity()
              (attention_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
              (ffn_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            )
            (2): ModuleDict(
              (attention): MultiheadAttention(
                (W_q): Linear(in_features=192, out_features=192, bias=True)
                (W_k): Linear(in_features=192, out_features=192, bias=True)
                (W_v): Linear(in_features=192, out_features=192, bias=True)
                (W_out): Linear(in_features=192, out_features=192, bias=True)
                (dropout): Dropout(p=0.2, inplace=False)
              )
              (ffn): FFN(
                (linear_first): Linear(in_features=192, out_features=384, bias=True)
                (activation): ReGLU()
                (dropout): Dropout(p=0.1, inplace=False)
                (linear_second): Linear(in_features=192, out_features=192, bias=True)
              )
              (attention_residual_dropout): Dropout(p=0.0, inplace=False)
              (ffn_residual_dropout): Dropout(p=0.0, inplace=False)
              (output): Identity()
              (attention_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
              (ffn_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            )
          )
          (head): Identity()
        )
        (head): Head(
          (normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (activation): ReLU()
          (linear): Linear(in_features=192, out_features=5, bias=True)
        )
      )
      (1): HFAutoModelForTextPrediction(
        (model): ElectraModel(
          (embeddings): ElectraEmbeddings(
            (word_embeddings): Embedding(30522, 768, padding_idx=0)
            (position_embeddings): Embedding(512, 768)
            (token_type_embeddings): Embedding(2, 768)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (encoder): ElectraEncoder(
            (layer): ModuleList(
              (0): ElectraLayer(
                (attention): ElectraAttention(
                  (self): ElectraSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): ElectraSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): ElectraIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                  (intermediate_act_fn): GELUActivation()
                )
                (output): ElectraOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (1): ElectraLayer(
                (attention): ElectraAttention(
                  (self): ElectraSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): ElectraSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): ElectraIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                  (intermediate_act_fn): GELUActivation()
                )
                (output): ElectraOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (2): ElectraLayer(
                (attention): ElectraAttention(
                  (self): ElectraSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): ElectraSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): ElectraIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                  (intermediate_act_fn): GELUActivation()
                )
                (output): ElectraOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (3): ElectraLayer(
                (attention): ElectraAttention(
                  (self): ElectraSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): ElectraSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): ElectraIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                  (intermediate_act_fn): GELUActivation()
                )
                (output): ElectraOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (4): ElectraLayer(
                (attention): ElectraAttention(
                  (self): ElectraSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): ElectraSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): ElectraIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                  (intermediate_act_fn): GELUActivation()
                )
                (output): ElectraOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (5): ElectraLayer(
                (attention): ElectraAttention(
                  (self): ElectraSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): ElectraSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): ElectraIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                  (intermediate_act_fn): GELUActivation()
                )
                (output): ElectraOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (6): ElectraLayer(
                (attention): ElectraAttention(
                  (self): ElectraSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): ElectraSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): ElectraIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                  (intermediate_act_fn): GELUActivation()
                )
                (output): ElectraOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (7): ElectraLayer(
                (attention): ElectraAttention(
                  (self): ElectraSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): ElectraSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): ElectraIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                  (intermediate_act_fn): GELUActivation()
                )
                (output): ElectraOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (8): ElectraLayer(
                (attention): ElectraAttention(
                  (self): ElectraSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): ElectraSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): ElectraIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                  (intermediate_act_fn): GELUActivation()
                )
                (output): ElectraOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (9): ElectraLayer(
                (attention): ElectraAttention(
                  (self): ElectraSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): ElectraSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): ElectraIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                  (intermediate_act_fn): GELUActivation()
                )
                (output): ElectraOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (10): ElectraLayer(
                (attention): ElectraAttention(
                  (self): ElectraSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): ElectraSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): ElectraIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                  (intermediate_act_fn): GELUActivation()
                )
                (output): ElectraOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (11): ElectraLayer(
                (attention): ElectraAttention(
                  (self): ElectraSelfAttention(
                    (query): Linear(in_features=768, out_features=768, bias=True)
                    (key): Linear(in_features=768, out_features=768, bias=True)
                    (value): Linear(in_features=768, out_features=768, bias=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                  (output): ElectraSelfOutput(
                    (dense): Linear(in_features=768, out_features=768, bias=True)
                    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                    (dropout): Dropout(p=0.1, inplace=False)
                  )
                )
                (intermediate): ElectraIntermediate(
                  (dense): Linear(in_features=768, out_features=3072, bias=True)
                  (intermediate_act_fn): GELUActivation()
                )
                (output): ElectraOutput(
                  (dense): Linear(in_features=3072, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
            )
          )
        )
        (head): Linear(in_features=768, out_features=5, bias=True)
      )
      (2): NumericalTransformer(
        (numerical_feature_tokenizer): NumEmbeddings(
          (layers): Sequential(
            (0): NumericalFeatureTokenizer()
            (1): ReLU()
          )
        )
        (cls_token): Identity()
        (transformer): FT_Transformer(
          (blocks): ModuleList(
            (0): ModuleDict(
              (attention): MultiheadAttention(
                (W_q): Linear(in_features=192, out_features=192, bias=True)
                (W_k): Linear(in_features=192, out_features=192, bias=True)
                (W_v): Linear(in_features=192, out_features=192, bias=True)
                (W_out): Linear(in_features=192, out_features=192, bias=True)
                (dropout): Dropout(p=0.2, inplace=False)
              )
              (ffn): FFN(
                (linear_first): Linear(in_features=192, out_features=384, bias=True)
                (activation): ReGLU()
                (dropout): Dropout(p=0.1, inplace=False)
                (linear_second): Linear(in_features=192, out_features=192, bias=True)
              )
              (attention_residual_dropout): Dropout(p=0.0, inplace=False)
              (ffn_residual_dropout): Dropout(p=0.0, inplace=False)
              (output): Identity()
              (ffn_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            )
            (1): ModuleDict(
              (attention): MultiheadAttention(
                (W_q): Linear(in_features=192, out_features=192, bias=True)
                (W_k): Linear(in_features=192, out_features=192, bias=True)
                (W_v): Linear(in_features=192, out_features=192, bias=True)
                (W_out): Linear(in_features=192, out_features=192, bias=True)
                (dropout): Dropout(p=0.2, inplace=False)
              )
              (ffn): FFN(
                (linear_first): Linear(in_features=192, out_features=384, bias=True)
                (activation): ReGLU()
                (dropout): Dropout(p=0.1, inplace=False)
                (linear_second): Linear(in_features=192, out_features=192, bias=True)
              )
              (attention_residual_dropout): Dropout(p=0.0, inplace=False)
              (ffn_residual_dropout): Dropout(p=0.0, inplace=False)
              (output): Identity()
              (attention_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
              (ffn_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            )
            (2): ModuleDict(
              (attention): MultiheadAttention(
                (W_q): Linear(in_features=192, out_features=192, bias=True)
                (W_k): Linear(in_features=192, out_features=192, bias=True)
                (W_v): Linear(in_features=192, out_features=192, bias=True)
                (W_out): Linear(in_features=192, out_features=192, bias=True)
                (dropout): Dropout(p=0.2, inplace=False)
              )
              (ffn): FFN(
                (linear_first): Linear(in_features=192, out_features=384, bias=True)
                (activation): ReGLU()
                (dropout): Dropout(p=0.1, inplace=False)
                (linear_second): Linear(in_features=192, out_features=192, bias=True)
              )
              (attention_residual_dropout): Dropout(p=0.0, inplace=False)
              (ffn_residual_dropout): Dropout(p=0.0, inplace=False)
              (output): Identity()
              (attention_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
              (ffn_normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            )
          )
          (head): Identity()
        )
        (head): Head(
          (normalization): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (activation): ReLU()
          (linear): Linear(in_features=192, out_features=5, bias=True)
        )
      )
      (3): TimmAutoModelForImagePrediction(
        (model): SwinTransformer(
          (patch_embed): PatchEmbed(
            (proj): Conv2d(3, 128, kernel_size=(4, 4), stride=(4, 4))
            (norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          )
          (layers): Sequential(
            (0): SwinTransformerStage(
              (downsample): Identity()
              (blocks): Sequential(
                (0): SwinTransformerBlock(
                  (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=128, out_features=384, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=128, out_features=128, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): Identity()
                  (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=128, out_features=512, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=512, out_features=128, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): Identity()
                )
                (1): SwinTransformerBlock(
                  (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=128, out_features=384, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=128, out_features=128, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.004)
                  (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=128, out_features=512, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=512, out_features=128, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.004)
                )
              )
            )
            (1): SwinTransformerStage(
              (downsample): PatchMerging(
                (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                (reduction): Linear(in_features=512, out_features=256, bias=False)
              )
              (blocks): Sequential(
                (0): SwinTransformerBlock(
                  (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=256, out_features=768, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=256, out_features=256, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.009)
                  (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=256, out_features=1024, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=1024, out_features=256, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.009)
                )
                (1): SwinTransformerBlock(
                  (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=256, out_features=768, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=256, out_features=256, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.013)
                  (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=256, out_features=1024, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=1024, out_features=256, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.013)
                )
              )
            )
            (2): SwinTransformerStage(
              (downsample): PatchMerging(
                (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (reduction): Linear(in_features=1024, out_features=512, bias=False)
              )
              (blocks): Sequential(
                (0): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.017)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.017)
                )
                (1): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.022)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.022)
                )
                (2): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.026)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.026)
                )
                (3): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.030)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.030)
                )
                (4): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.035)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.035)
                )
                (5): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.039)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.039)
                )
                (6): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.043)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.043)
                )
                (7): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.048)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.048)
                )
                (8): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.052)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.052)
                )
                (9): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.057)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.057)
                )
                (10): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.061)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.061)
                )
                (11): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.065)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.065)
                )
                (12): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.070)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.070)
                )
                (13): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.074)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.074)
                )
                (14): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.078)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.078)
                )
                (15): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.083)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.083)
                )
                (16): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.087)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.087)
                )
                (17): SwinTransformerBlock(
                  (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=512, out_features=1536, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=512, out_features=512, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.091)
                  (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=512, out_features=2048, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=2048, out_features=512, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.091)
                )
              )
            )
            (3): SwinTransformerStage(
              (downsample): PatchMerging(
                (norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
                (reduction): Linear(in_features=2048, out_features=1024, bias=False)
              )
              (blocks): Sequential(
                (0): SwinTransformerBlock(
                  (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=1024, out_features=3072, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=1024, out_features=1024, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.096)
                  (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=1024, out_features=4096, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=4096, out_features=1024, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.096)
                )
                (1): SwinTransformerBlock(
                  (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                  (attn): WindowAttention(
                    (qkv): Linear(in_features=1024, out_features=3072, bias=True)
                    (attn_drop): Dropout(p=0.0, inplace=False)
                    (proj): Linear(in_features=1024, out_features=1024, bias=True)
                    (proj_drop): Dropout(p=0.0, inplace=False)
                    (softmax): Softmax(dim=-1)
                  )
                  (drop_path1): DropPath(drop_prob=0.100)
                  (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                  (mlp): Mlp(
                    (fc1): Linear(in_features=1024, out_features=4096, bias=True)
                    (act): GELU()
                    (drop1): Dropout(p=0.0, inplace=False)
                    (norm): Identity()
                    (fc2): Linear(in_features=4096, out_features=1024, bias=True)
                    (drop2): Dropout(p=0.0, inplace=False)
                  )
                  (drop_path2): DropPath(drop_prob=0.100)
                )
              )
            )
          )
          (norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (head): ClassifierHead(
            (global_pool): SelectAdaptivePool2d(pool_type=avg, flatten=Identity())
            (drop): Dropout(p=0.0, inplace=False)
            (fc): Identity()
            (flatten): Identity()
          )
        )
        (head): Linear(in_features=1024, out_features=5, bias=True)
      )
    )
    (adapter): ModuleList(
      (0): Linear(in_features=192, out_features=1024, bias=True)
      (1): Linear(in_features=768, out_features=1024, bias=True)
      (2): Linear(in_features=192, out_features=1024, bias=True)
      (3): Linear(in_features=1024, out_features=1024, bias=True)
    )
    (fusion_mlp): Sequential(
      (0): MLP(
        (layers): Sequential(
          (0): Unit(
            (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (fc): Linear(in_features=4096, out_features=128, bias=True)
            (act_fn): LeakyReLU(negative_slope=0.01)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (head): Linear(in_features=128, out_features=5, bias=True)
  )
)
Predicting DataLoader 0: 100%
47/47 [00:22<00:00, 2.09it/s]
MultiModalFeaturePreprocessor(column_types={'AdoptionSpeed': 'categorical',
                                            'Age': 'numerical',
                                            'Breed1': 'numerical',
                                            'Breed2': 'numerical',
                                            'Color1': 'numerical',
                                            'Color2': 'numerical',
                                            'Color3': 'numerical',
                                            'Description': 'text',
                                            'Dewormed': 'categorical',
                                            'Fee': 'numerical',
                                            'FurLength': 'categorical',
                                            'Gender': 'categorical',
                                            'Health': 'categorical',
                                            'Images': 'image_p...
                              config={'image': {'missing_value_strategy': 'zero'}, 'text': {'normalize_text': False}, 'categorical': {'minimum_cat_count': 100, 'maximum_num_cat': 20, 'convert_to_text': False}, 'numerical': {'convert_to_text': False, 'scaler_with_mean': True, 'scaler_with_std': True}, 'document': {'missing_value_strategy': 'zero'}, 'label': {'numerical_label_preprocessing': 'standardscaler'}, 'pos_label': None, 'mixup': {'turn_on': False, 'mixup_alpha': 0.8, 'cutmix_alpha': 1.0, 'cutmix_minmax': None, 'prob': 1.0, 'switch_prob': 0.5, 'mode': 'batch', 'turn_off_epoch': 5, 'label_smoothing': 0.1}, 'templates': {'turn_on': False, 'num_templates': 30, 'template_length': 2048, 'preset_templates': ['super_glue', 'rte'], 'custom_templates': None}, 'video': {'missing_value_strategy': 'zero', 'requires_column_info': False}},
                              label_column='AdoptionSpeed',
                              label_generator=<autom3l.multimodal.data.label_encoder.CustomLabelEncoder object at 0x7ff296362f40>)
[0 0 0 0 0 0 0 0 0 0]
[2 2 4 2 3 3 4 3 4 2]
