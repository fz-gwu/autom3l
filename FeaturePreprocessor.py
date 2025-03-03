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
                              label_generator=<autom3l.multimodal.data.label_encoder.CustomLabelEncoder object at 0x7993d035f2b0>)
