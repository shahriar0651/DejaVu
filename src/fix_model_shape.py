import torch

# ckpt_path = 'checkpoints/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth'
# save_path = 'checkpoints/mvxnet_fpn_dv_second_secfpn_fixed.pth'

ckpt_path = '../mmdetection3d/checkpoints/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth'
save_path = '../mmdetection3d/checkpoints/mvxnet_fpn_dv_second_secfpn_fixed.pth'

# import torch

# ckpt_path = 'checkpoints/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth'
# save_path = 'checkpoints/mvxnet_fpn_dv_second_secfpn_fixed_final.pth'

ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

def fix_shape(tensor, key):
    if tensor.ndim != 5:
        return tensor
    if tensor.shape[0] == 3 and tensor.shape[2:] == (3, 3):  # pattern: [3, in, 3, 3, out]
        print(f"Fixing {key}: {tensor.shape} → ", end="")
        fixed = tensor.permute(0, 2, 3, 1, 4).contiguous()  # [3, in, 3, 3, out] → [3, 3, 3, in, out]
        print(f"{fixed.shape}")
        return fixed
    return tensor

# Apply fix to relevant layers
for key in list(state_dict.keys()):
    if 'pts_middle_encoder' in key and '.weight' in key:
        state_dict[key] = fix_shape(state_dict[key], key)

# Save updated checkpoint
if 'state_dict' in ckpt:
    ckpt['state_dict'] = state_dict
torch.save(ckpt, save_path)

print(f"\n✅ Fixed model saved to: {save_path}")


# ckpt = torch.load(ckpt_path, map_location='cpu')
# state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

# def fix_conv3d_weight(w, key):
#     if w.ndim != 5:
#         print(f"[SKIP] {key} is not 5D")
#         return w

#     # Case: [O, 3, I, K, K] → [K, K, K, I, O]
#     if w.shape[1] == 3:
#         print(f"[FIX] {key}: {w.shape} → ", end='')
#         fixed = w.permute(3, 4, 1, 2, 0).contiguous()
#         print(f"{fixed.shape}")
#         return fixed
#     return w

# # Fix known bad keys
# for k in list(state_dict.keys()):
#     if 'pts_middle_encoder' in k and '.weight' in k:
#         state_dict[k] = fix_conv3d_weight(state_dict[k], k)

# # Save
# if 'state_dict' in ckpt:
#     ckpt['state_dict'] = state_dict
# else:
#     ckpt = state_dict

# torch.save(ckpt, save_path)
# print(f"\n✅ Fixed checkpoint saved to: {save_path}")


# import torch

# # Path to original and fixed checkpoints
# input_path = '../mmdetection3d/checkpoints/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth'
# output_path = '../mmdetection3d/checkpoints/mvxnet_fpn_dv_second_secfpn_fixed.pth'

# # Load checkpoint
# checkpoint = torch.load(input_path, map_location='cpu')
# state_dict = checkpoint.get('state_dict', checkpoint)

# # List of keys to fix manually
# keys_to_fix = [
#     'pts_middle_encoder.conv_input.0.weight',
#     'pts_middle_encoder.conv_out.0.weight',
#     'pts_middle_encoder.encoder_layers.encoder_layer1.0.0.weight',
#     'pts_middle_encoder.encoder_layers.encoder_layer2.0.0.weight',
#     'pts_middle_encoder.encoder_layers.encoder_layer2.1.0.weight',
#     'pts_middle_encoder.encoder_layers.encoder_layer2.2.0.weight',
#     'pts_middle_encoder.encoder_layers.encoder_layer3.0.0.weight',
#     'pts_middle_encoder.encoder_layers.encoder_layer3.1.0.weight',
#     'pts_middle_encoder.encoder_layers.encoder_layer3.2.0.weight',
#     'pts_middle_encoder.encoder_layers.encoder_layer4.0.0.weight',
#     'pts_middle_encoder.encoder_layers.encoder_layer4.1.0.weight',
#     'pts_middle_encoder.encoder_layers.encoder_layer4.2.0.weight',
# ]

# def fix_shape(tensor, key):
#     # All of these seem to need: [O, I, D, H, W] ← [D, H, W, I, O]
#     if tensor.ndim != 5:
#         print(f"[SKIP] {key} is not 5D")
#         return tensor

#     fixed_tensor = tensor.permute(4, 3, 0, 1, 2).contiguous()
#     print(f"[FIXED] {key}: {tensor.shape} → {fixed_tensor.shape}")
#     return fixed_tensor

# # Apply fixes
# for key in keys_to_fix:
#     if key in state_dict:
#         state_dict[key] = fix_shape(state_dict[key], key)
#     else:
#         print(f"[MISSING] {key} not found in checkpoint.")

# # Save updated checkpoint
# if 'state_dict' in checkpoint:
#     checkpoint['state_dict'] = state_dict
#     torch.save(checkpoint, output_path)
# else:
#     torch.save(state_dict, output_path)

# print(f"\n✅ Fixed checkpoint saved to: {output_path}")


# import torch

# def fix_tensor_shape(tensor):
#     """
#     Try to reshape 5D convolution weights to [out, in, D, H, W]
#     """
#     if tensor.ndim != 5:
#         return tensor  # Do nothing if not 5D

#     shape = tensor.shape
#     # Common wrong shape: [D, H, W, in, out] → [out, in, D, H, W]
#     if shape[0] <= 5 and shape[-1] <= 128:  # e.g., [3, 3, 3, 64, 16]
#         fixed = tensor.permute(4, 3, 0, 1, 2)  # [O, I, D, H, W]
#         return fixed
#     # Common wrong shape: [out, D, H, W, in]
#     elif shape[1] <= 5:
#         fixed = tensor.permute(0, 4, 1, 2, 3)
#         return fixed
#     else:
#         # Leave as is if unsure
#         return tensor

# def fix_checkpoint_shapes(input_path, output_path):
#     checkpoint = torch.load(input_path, map_location='cpu')
#     state_dict = checkpoint.get('state_dict', checkpoint)

#     fixed_state_dict = {}
#     for key, tensor in state_dict.items():
#         if isinstance(tensor, torch.Tensor):
#             fixed_tensor = fix_tensor_shape(tensor)
#             if fixed_tensor.shape != tensor.shape:
#                 print(f"[FIXED] {key}: {tensor.shape} → {fixed_tensor.shape}")
#             fixed_state_dict[key] = fixed_tensor
#         else:
#             fixed_state_dict[key] = tensor

#     # Save
#     if 'state_dict' in checkpoint:
#         checkpoint['state_dict'] = fixed_state_dict
#         torch.save(checkpoint, output_path)
#     else:
#         torch.save(fixed_state_dict, output_path)

#     print(f"\n✅ Fixed checkpoint saved to: {output_path}")

# # === Run ===
# if __name__ == '__main__':
#     input_ckpt = '../mmdetection3d/checkpoints/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth'
#     output_ckpt = '../mmdetection3d/checkpoints/mvxnet_fpn_dv_second_secfpn_fixed.pth'
#     fix_checkpoint_shapes(input_ckpt, output_ckpt)


# import torch

# path = '../mmdetection3d/checkpoints/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth'
# model = torch.load(path)

# for key in model['state_dict'].keys():
#     # Match all 3D convolution layers in encoder
#     if 'pts_middle_encoder.encoder_layers.encoder_layer' in key and key.endswith('weight'):
#         if model['state_dict'][key].ndim == 5:
#             # Convert from [out, D, H, W, in] → [out, in, D, H, W]
#             model['state_dict'][key] = model['state_dict'][key].permute(0, 4, 1, 2, 3)

# torch.save(model, "../mmdetection3d/checkpoints/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a_fixed.pth")


# import torch

# path = '../mmdetection3d/checkpoints/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a.pth'
# model = torch.load(path)

# for key in model['state_dict'].keys():
#     if (key == 'pts_middle_encoder.conv_input.0.weight') or (key == 'pts_middle_encoder.conv_input.0.weight') or (key == 'pts_middle_encoder.conv_out.0.weight') or (key == 'pts_middle_encoder.encoder_layers.encoder_layer3.2.0.weight') or (key == 'pts_middle_encoder.encoder_layers.encoder_layer2.2.0.weight') or (key == 'pts_middle_encoder.encoder_layers.encoder_layer1.2.0.weight') or (('pts_middle_encoder.encoder_layers.encoder_layer' in key) and ('conv' in key)):
#         model['state_dict'][key] = torch.transpose(model['state_dict'][key],0,1)
#         model['state_dict'][key] = torch.transpose(model['state_dict'][key],1,2)
#         model['state_dict'][key] = torch.transpose(model['state_dict'][key],2,3)
#         model['state_dict'][key] = torch.transpose(model['state_dict'][key],3,4)


# torch.save(model, "../mmdetection3d/checkpoints/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class-8963258a_fixed.pth")