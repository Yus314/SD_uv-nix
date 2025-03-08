import torch


def create_mask(height, width, mask_size, latent_mask_min, latent_mask_max, device):
    mask = torch.zeros(1, 4, height // 8, width // 8).to(device)
    mask[:, :, latent_mask_min:latent_mask_max, latent_mask_min:latent_mask_max] = 0
    return mask
