import torch
import re
import os
from collections import OrderedDict

def modify_state_dict(input_file, output_file):
    # Load the checkpoint
    checkpoint = torch.load(input_file, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Define model mappings
    model_mappings = {
        'model.base_models.categorical_transformer': 'model.0',
        'model.base_models.hf_text-electra': 'model.1',
        'model.base_models.numerical_transformer': 'model.2',
        'model.base_models.timm_image-swin_transformer': 'model.3'
    }
    
    # Define adapter and fusion key mappings (from name 2 to name 1)
    adapter_fusion_mappings = {
        'model.linear_layers.hf_text-electra.weight': 'adapter.0.weight',
        'model.linear_layers.hf_text-electra.bias': 'adapter.0.bias',
        'model.linear_layers.numerical_transformer.weight': 'adapter.1.weight',
        'model.linear_layers.numerical_transformer.bias': 'adapter.1.bias',
        'model.linear_layers.categorical_transformer.weight': 'adapter.2.weight',
        'model.linear_layers.categorical_transformer.bias': 'adapter.2.bias',
        'model.linear_layers.timm_image-swin_transformer.weight': 'adapter.3.weight',
        'model.linear_layers.timm_image-swin_transformer.bias': 'adapter.3.bias',
        'model.fusion_model.0.weight': 'fusion_mlp.0.layers.0.norm.weight',
        'model.fusion_model.0.bias': 'fusion_mlp.0.layers.0.norm.bias',
        'model.fusion_model.2.weight': 'fusion_mlp.0.layers.0.fc.weight',
        'model.fusion_model.2.bias': 'fusion_mlp.0.layers.0.fc.bias',
        'model.fusion_head.weight': 'head.weight',
        'model.fusion_head.bias': 'head.bias'
    }
    
    # Create a new state dictionary with the modified keys
    new_state_dict = OrderedDict()
    
    for key, value in state_dict.items():
        new_key = key
        
        # Handle base model renaming
        for old_prefix, new_prefix in model_mappings.items():
            if key.startswith(old_prefix):
                # Extract the part after the base model prefix
                suffix = key[len(old_prefix):]
                new_key = f"{new_prefix}{suffix}"
                break
                
        # If key wasn't modified by model mappings, check adapter and fusion keys
        if new_key == key:  # Only apply second transformation if first wasn't applied
            if key in adapter_fusion_mappings:
                new_key = adapter_fusion_mappings[key]
        
        new_state_dict[new_key] = value
    
    # Prepare the output checkpoint
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint['state_dict'] = new_state_dict
        output_checkpoint = checkpoint
    else:
        output_checkpoint = new_state_dict
    
    # Save the modified checkpoint
    torch.save(output_checkpoint, output_file)
    print(f"Modified checkpoint saved to {output_file}")
    
    # Print some statistics
    print(f"Original keys count: {len(state_dict)}")
    print(f"Modified keys count: {len(new_state_dict)}")
    
    # Print the first 10 keys of the new state dictionary
    print("\nFirst 10 keys of the modified state dictionary:")
    for i, key in enumerate(list(new_state_dict.keys())[:10]):
        print(f"{i+1}. {key}")
    
    # Print some transformed key examples for verification
    if len(state_dict) > 0:
        print("\nExample key transformations:")
        count = 0
        for old_key, new_key in zip(state_dict.keys(), new_state_dict.keys()):
            if old_key != new_key:
                print(f"Old: {old_key}")
                print(f"New: {new_key}")
                print()
                count += 1
                if count >= 5:
                    break

if __name__ == "__main__":
    # Specify the exact input and output paths
    input_file = "/home/ubuntu/autom3l/autom3l_code/output/petfinder_k1_llm_20250302_234413/model.ckpt"
    output_file = "/home/ubuntu/autom3l/autom3l_code/output/test/model.ckpt"
    
    # Verify input file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    modify_state_dict(input_file, output_file)
