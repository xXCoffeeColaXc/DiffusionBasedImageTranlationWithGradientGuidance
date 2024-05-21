import torch
import os.path as osp
import torch.nn.functional as F
import argparse


# def load_backbone(path: str):
#     if not osp.isfile(path):
#         raise FileNotFoundError(
#             f"{path} dont exist(absolute path: {osp.abspath(path)})"
#         )
#     weight = torch.load(path, map_location="cpu")
#     weight["pos_embed"] = torch.cat(
#         (
#             weight["pos_embed"][:, :1, :],
#             F.interpolate(
#                 weight["pos_embed"][:, 1:, :]
#                 .reshape(1, 37, 37, 1024)
#                 .permute(0, 3, 1, 2),
#                 size=(32, 32),
#                 mode="bicubic",
#                 align_corners=False,
#             )
#             .permute(0, 2, 3, 1)
#             .reshape(1, 1024, 1024),
#         ),
#         dim=1,
#     )
#     weight["patch_embed.proj.weight"] = F.interpolate(
#         weight["patch_embed.proj.weight"].float(),
#         size=(16, 16),
#         mode="bicubic",
#         align_corners=False,
#     )
#     return weight

def load_backbone(path: str):
    if not osp.isfile(path):
        raise FileNotFoundError(f"{path} don't exist (absolute path: {osp.abspath(path)})")
    weight = torch.load(path, map_location="cpu")
    
    # Debug: Print the shape of the positional embedding tensor
    print("Original pos_embed shape:", weight["pos_embed"].shape)

    # Assuming the position embedding is a flat vector that needs to be split into a grid-like structure
    if weight["pos_embed"].shape == (1, 512):  # Adjust this condition based on your print statement
        grid_size = 32  # This is an assumption; adjust based on actual model architecture
        weight["pos_embed"] = torch.cat((
            weight["pos_embed"][:, :1],
            F.interpolate(
                weight["pos_embed"][:, 1:].reshape(1, grid_size, grid_size, -1).permute(0, 3, 1, 2),
                size=(grid_size, grid_size),
                mode="bicubic",
                align_corners=False
            ).permute(0, 2, 3, 1).flatten(1)
        ), dim=1)
    
    # Further processing if needed
    return weight

def main(args):
    dinov2_segmentor_path = args.dinov2_segmentor_path
    backbone = args.backbone
    rein_head = args.rein_head

    if not osp.isfile(dinov2_segmentor_path):
        weight = torch.load(rein_head, map_location='cpu')
        weight['state_dict'].update({f'backbone.{k}': v for k, v in load_backbone(backbone).items()})
        torch.save(weight, dinov2_segmentor_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and process model weights.")
    parser.add_argument("--dinov2_segmentor_path", required=True, help="Path to the DINOv2 segmentor checkpoint")
    parser.add_argument("--backbone", required=True, help="Path to the DINOv2 backbone weights")
    parser.add_argument("--rein_head", required=True, help="Path to the REIN weights")
    
    args = parser.parse_args()
    main(args)
