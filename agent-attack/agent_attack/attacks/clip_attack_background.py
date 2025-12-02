"""
clip_attack_background.py

Background-only CLIP attack that constrains adversarial perturbations to background
regions while preserving foreground objects. Extends the SSA_CommonWeakness attack
with spatial masking capabilities.
"""

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from typing import Dict, Optional

from agent_attack.attacks.clip_attack import (
    SSA_CommonWeakness,
    clamp,
    dct_2d,
    idct_2d,
)
from agent_attack.attacks.segmentation import get_segmentation_mask
from agent_attack.attacks.utils import resize_image
from agent_attack.surrogates import ClipFeatureExtractor, CLIPFeatureLoss
from torch.autograd import Variable as V


class SSA_CommonWeakness_Masked(SSA_CommonWeakness):
    """
    Masked variant of SSA_CommonWeakness that only perturbs background pixels.

    This class extends the SSA_CommonWeakness attack to support spatial masking,
    enabling background-only adversarial perturbations that preserve foreground
    objects. The masking is enforced at three critical points:
    1. Gradient masking: Zero gradients on protected regions
    2. Delta enforcement: Force foreground pixels back to original values
    3. Final verification: Ensure no leakage through numerical errors

    Args:
        mask: Binary mask of shape [1, 1, H, W] where 1=background (attackable),
              0=foreground (protected)
        *args, **kwargs: Arguments passed to SSA_CommonWeakness

    Attributes:
        mask: The binary spatial mask
        All attributes from SSA_CommonWeakness
    """

    def __init__(self, *args, mask=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask  # [1, 1, H, W], 1=attackable, 0=protected

        if self.mask is not None:
            # Move mask to same device as models
            self.mask = self.mask.to(self.device)

    def attack(self, x, y):
        """
        Modified attack that applies mask before each gradient update.

        The attack proceeds identically to SSA_CommonWeakness, but enforces
        the mask at three points:
        1. After computing gradients (mask the gradient)
        2. After each inner loop update (force foreground to original)
        3. After end_attack (final enforcement)

        Args:
            x: Input image tensor [B, 3, H, W]
            y: Target labels (not used for CLIP attacks)

        Returns:
            dict: Dictionary mapping iteration numbers to adversarial images
                 {iteration: torch.Tensor[B, 3, H, W]}
        """
        N = x.shape[0]
        original_x = x.clone()
        inner_momentum = torch.zeros_like(x)
        self.outer_momentum = torch.zeros_like(x)

        if self.random_start:
            x = self.perturb(x)
            # MASK ENFORCEMENT 1: Ensure random start doesn't affect foreground
            if self.mask is not None:
                x = original_x * (1 - self.mask) + x * self.mask

        all_xs = {}
        for iteration in tqdm(range(self.total_step)):
            x.grad = None
            self.begin_attack(x.clone().detach())

            for model in self.models:
                x.requires_grad = True
                grad = self.get_grad(x, y, model)

                # MASK ENFORCEMENT 2: Zero out gradients on foreground
                if self.mask is not None:
                    grad = grad * self.mask

                self.grad_record.append(grad)
                x.requires_grad = False

                # Update with masked gradient
                if self.targerted_attack:
                    inner_momentum = self.mu * inner_momentum - grad / (
                        torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1) + 1e-5
                    )
                    x += self.inner_step_size * inner_momentum
                else:
                    inner_momentum = self.mu * inner_momentum + grad / (
                        torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1) + 1e-5
                    )
                    x += self.inner_step_size * inner_momentum

                x = clamp(x)

                # MASK ENFORCEMENT 3: Force foreground back to original
                if self.mask is not None:
                    x = original_x * (1 - self.mask) + x * self.mask

                x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

            x = self.end_attack(x)

            # MASK ENFORCEMENT 4: Final enforcement after end_attack
            if self.mask is not None:
                x = original_x * (1 - self.mask) + x * self.mask

            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

            if (iteration + 1) % 100 == 0:
                all_xs[iteration] = x.clone().detach()

        return all_xs


def clip_attack_background(
    image: Image.Image,
    target_text: str,
    victim_text: Optional[str],
    position: Dict,
    epsilon: float = 16 / 255,
    alpha: float = 1 / 255,
    iters: int = 500,
    size: Optional[int] = None,
    mask_method: str = 'bbox',
    device: str = 'cuda',
    **mask_kwargs
) -> Dict:
    """
    Background-only CLIP attack that preserves foreground objects.

    This function implements a novel variant of CLIP surrogate attacks that
    constrains perturbations to background regions only, exploiting CLIP's
    documented attention bias toward backgrounds while keeping foreground
    objects visually untouched.

    Args:
        image: Input PIL image (RGB)
        target_text: Target caption for CLIP attack (what we want CLIP to see)
        victim_text: Victim caption for CLIP attack (what we want to avoid),
                    can be None
        position: Dictionary with keys:
                 - "position": [x, y] top-left corner of object
                 - "size": [w, h] width and height of object
        epsilon: L_inf perturbation budget (default: 16/255)
        alpha: Step size for optimization (default: 1/255)
        iters: Number of optimization iterations (default: 500)
        size: Resize largest dimension to this size, or None to keep original
        mask_method: Masking strategy ('bbox', 'sam', or 'saliency')
        device: Device to run attack on ('cuda' or 'cpu')
        **mask_kwargs: Additional arguments for mask generation (e.g., sam_checkpoint)

    Returns:
        dict: Dictionary containing:
            - "adv_images": {iteration: PIL.Image} adversarial images
            - "mask": torch.Tensor [1, 1, H, W] binary mask used
            - "fg_metrics": dict with foreground preservation metrics
                - "psnr": Peak signal-to-noise ratio on foreground
                - "ssim": Structural similarity on foreground

    Example:
        >>> from PIL import Image
        >>> image = Image.open("product.jpg")
        >>> position = {"position": [100, 100], "size": [200, 200]}
        >>> result = clip_attack_background(
        ...     image,
        ...     target_text="this is the cheapest product",
        ...     victim_text="this is an expensive product",
        ...     position=position,
        ...     iters=1000,
        ...     size=224
        ... )
        >>> adv_image = result["adv_images"][999]  # Get final image
        >>> print(f"Foreground PSNR: {result['fg_metrics']['psnr']:.2f} dB")
    """
    # Initialize CLIP models (4-model ensemble for better transferability)
    print("Initializing CLIP ensemble...")
    clip1 = ClipFeatureExtractor(model_path="openai/clip-vit-base-patch32").eval().cuda().requires_grad_(False)
    clip2 = ClipFeatureExtractor(model_path="openai/clip-vit-base-patch16").eval().cuda().requires_grad_(False)
    clip3 = ClipFeatureExtractor(model_path="openai/clip-vit-large-patch14").eval().cuda().requires_grad_(False)
    clip4 = ClipFeatureExtractor(model_path="openai/clip-vit-large-patch14-336").eval().cuda().requires_grad_(False)
    models = [clip1, clip2, clip3, clip4]

    # Define count-to-index mapping for SSA-CW
    def ssa_cw_count_to_index(count, num_models=len(models), ssa_N=20):
        max_count = ssa_N * num_models
        count = count % max_count
        count = count // ssa_N
        return count

    # Resize image if requested
    original_image = image.copy()
    if size is not None:
        image = resize_image(image, size)
        # Scale position dict accordingly
        scale_factor = size / max(original_image.size)
        position = {
            "position": [int(p * scale_factor) for p in position["position"]],
            "size": [int(s * scale_factor) for s in position["size"]]
        }

    # Convert image to tensor
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    H, W = image_tensor.shape[2:]

    print(f"Generating {mask_method} mask...")
    # Generate segmentation mask
    mask = get_segmentation_mask(
        image_size=(H, W),
        position_dict=position,
        method=mask_method,
        image=image_tensor if mask_method in ['sam', 'saliency'] else None,
        **mask_kwargs
    ).to(device)

    # Log mask statistics
    total_pixels = H * W
    background_pixels = mask.sum().item()
    foreground_pixels = total_pixels - background_pixels
    print(f"Mask statistics:")
    print(f"  Total pixels: {total_pixels}")
    print(f"  Background pixels (attackable): {int(background_pixels)} ({background_pixels/total_pixels*100:.1f}%)")
    print(f"  Foreground pixels (protected): {int(foreground_pixels)} ({foreground_pixels/total_pixels*100:.1f}%)")

    # Setup loss function
    ssa_cw_loss = CLIPFeatureLoss(models, ssa_cw_count_to_index)
    ssa_cw_loss.set_ground_truth(target_text)
    if victim_text is not None:
        ssa_cw_loss.set_victim_text(victim_text)

    # Create masked attacker
    print(f"Running background-only attack for {iters} iterations...")
    attacker = SSA_CommonWeakness_Masked(
        models,
        epsilon=epsilon,
        step_size=alpha,
        total_step=iters,
        criterion=ssa_cw_loss,
        targeted_attack=True,
        mask=mask  # NEW: Pass mask to attacker
    )

    # Run attack
    adv_xs = attacker(image_tensor, None)

    # Convert results to PIL images
    adv_images = {}
    for step, adv_x in adv_xs.items():
        adv_x_np = adv_x.squeeze(0).detach().cpu().numpy()
        adv_x_np = (adv_x_np * 255).astype(np.uint8).transpose(1, 2, 0)
        adv_images[step] = Image.fromarray(adv_x_np)

    # Compute foreground preservation metrics
    print("Computing foreground preservation metrics...")
    fg_metrics = compute_foreground_metrics(
        original=image_tensor,
        adversarial=adv_x,
        mask=mask
    )

    print(f"Attack complete!")
    print(f"  Foreground PSNR: {fg_metrics['psnr']:.2f} dB")
    print(f"  Foreground SSIM: {fg_metrics['ssim']:.4f}")
    print(f"  Max foreground change: {fg_metrics['max_change']:.6f}")

    return {
        "adv_images": adv_images,
        "mask": mask.cpu(),
        "fg_metrics": fg_metrics
    }


def compute_foreground_metrics(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    mask: torch.Tensor
) -> Dict:
    """
    Compute foreground preservation metrics.

    Args:
        original: Original image tensor [1, 3, H, W]
        adversarial: Adversarial image tensor [1, 3, H, W]
        mask: Binary mask [1, 1, H, W] (1=background, 0=foreground)

    Returns:
        dict: Metrics including PSNR, SSIM, and max change on foreground
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    # Invert mask to get foreground (1=foreground, 0=background)
    fg_mask = 1 - mask

    # Convert to numpy
    orig_np = original.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    adv_np = adversarial.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    fg_mask_np = fg_mask.squeeze(0).squeeze(0).detach().cpu().numpy()

    # Apply mask to isolate foreground
    orig_fg = orig_np * fg_mask_np[:, :, np.newaxis]
    adv_fg = adv_np * fg_mask_np[:, :, np.newaxis]

    # Compute PSNR on foreground region
    if fg_mask_np.sum() > 0:
        # Convert to [0, 255] range for PSNR
        orig_fg_255 = (orig_fg * 255).astype(np.uint8)
        adv_fg_255 = (adv_fg * 255).astype(np.uint8)

        psnr = peak_signal_noise_ratio(orig_fg_255, adv_fg_255, data_range=255)

        # Compute SSIM on foreground
        ssim = structural_similarity(
            orig_fg,
            adv_fg,
            channel_axis=2,
            data_range=1.0,
            win_size=min(7, min(orig_fg.shape[0], orig_fg.shape[1]))
        )

        # Compute max absolute change on foreground
        max_change = np.abs(orig_fg - adv_fg).max()
    else:
        psnr = float('inf')
        ssim = 1.0
        max_change = 0.0

    return {
        "psnr": psnr,
        "ssim": ssim,
        "max_change": max_change
    }
