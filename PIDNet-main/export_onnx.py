# ------------------------------------------------------------------------------
# ONNX Export Script for PIDNet
# ------------------------------------------------------------------------------

import argparse
import os
import torch
import _init_paths
import models

def parse_args():
    parser = argparse.ArgumentParser(description='Export PIDNet to ONNX')
    
    parser.add_argument('--model', 
                        help='Model type: pidnet-s, pidnet-m, or pidnet-l',
                        default='pidnet-s',
                        type=str)
    parser.add_argument('--checkpoint',
                        help='Path to model checkpoint (.pt file)',
                        default='../Cityscapes/PIDNet_S_Simple_Cityscapes_val.pt',
                        type=str)
    parser.add_argument('--output',
                        help='Output ONNX file path',
                        default=None,
                        type=str)
    parser.add_argument('--num-classes',
                        help='Number of classes (19 for Cityscapes, 11 for CamVid)',
                        default=19,
                        type=int)
    parser.add_argument('--input-size',
                        help='Input image size [height, width]',
                        nargs=2,
                        default=[1024, 2048],
                        type=int)
    parser.add_argument('--opset-version',
                        help='ONNX opset version',
                        default=11,
                        type=int)
    parser.add_argument('--dynamic',
                        help='Export with dynamic batch size and image size',
                        action='store_true')

    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = f"{base_name}.onnx"
    
    return args

def load_pretrained(model, pretrained):
    """Load pretrained weights"""
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    # Remove 'model.' prefix if present
    pretrained_dict = {k[6:] if k.startswith('model.') else k: v 
                      for k, v in pretrained_dict.items() 
                      if (k[6:] if k.startswith('model.') else k) in model_dict 
                      and v.shape == model_dict[k[6:] if k.startswith('model.') else k].shape}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    return model

def main():
    args = parse_args()
    
    print(f"Loading model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"Input size: {args.input_size}")
    print(f"Number of classes: {args.num_classes}")
    
    # Create model
    model = models.pidnet.get_pred_model(args.model, args.num_classes)
    
    # Load pretrained weights
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        # Try relative paths
        possible_paths = [
            checkpoint_path,
            os.path.join('..', checkpoint_path),
            os.path.join('Cityscapes', os.path.basename(checkpoint_path)),
            os.path.join('..', 'Cityscapes', os.path.basename(checkpoint_path))
        ]
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        model = load_pretrained(model, checkpoint_path)
    else:
        print(f"Warning: Checkpoint file not found: {checkpoint_path}")
        print("Exporting model without pretrained weights...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Move to CPU for ONNX export (ONNX export works better on CPU)
    device = torch.device('cpu')
    model = model.to(device)
    
    # Create dummy input
    height, width = args.input_size
    dummy_input = torch.randn(1, 3, height, width).to(device)
    
    print(f"\nExporting to ONNX...")
    print(f"Input shape: {dummy_input.shape}")
    
    # Export to ONNX - 使用和 2onnx.py 相同的簡單方式
    try:
        # Set environment variable to handle encoding
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # 直接使用原始模型，不使用 trace（因為新版本 PyTorch 不支持 trace 的 ScriptModule）
        model_to_export = model
        
        # 重定向 stdout 來避免編碼問題
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding='utf-8', errors='replace')
        
        try:
            if args.dynamic:
                # Dynamic export with variable batch size and image dimensions
                torch.onnx.export(
                    model_to_export,
                    dummy_input,
                    args.output,
                    input_names=['input'],
                    output_names=['output'],
                    opset_version=args.opset_version,
                    do_constant_folding=True,
                    dynamic_axes={
                        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
                    }
                )
            else:
                # Static export - 直接使用簡單的方式，和 2onnx.py 一樣
                torch.onnx.export(
                    model_to_export,
                    dummy_input,
                    args.output,
                    input_names=['input'],
                    output_names=['output'],
                    opset_version=args.opset_version,
                    do_constant_folding=True,
                )
        finally:
            # 恢復 stdout
            sys.stdout = old_stdout
        
        print("Exported successfully")
        
        print(f"\nSuccessfully exported to: {args.output}")
        
        # Verify the exported model
        try:
            import onnx
            onnx_model = onnx.load(args.output)
            onnx.checker.check_model(onnx_model)
            actual_opset = onnx_model.opset_import[0].version
            print(f"ONNX model verification passed! (opset version: {actual_opset})")
        except ImportError:
            print("Note: onnx package not found, skipping model verification")
        except Exception as e:
            print(f"Warning: ONNX model verification failed: {e}")
            
    except Exception as e:
        print(f"\nExport failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
