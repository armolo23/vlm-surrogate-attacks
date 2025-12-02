"""
segmentation.py

Segmentation mask generation for background-only adversarial attacks.
Supports multiple masking strategies: bounding box, SAM, and saliency-based.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


def bbox_mask(image_size: Tuple[int, int], position_dict: Dict) -> torch.Tensor:
    """
    Create binary mask from bounding box.

    This is the simplest masking strategy that protects a rectangular region
    defined by the position dictionary. All pixels outside this region are
    marked as attackable (background), while pixels inside are protected (foreground).

    Args:
        image_size: (H, W) tuple specifying image dimensions
        position_dict: Dictionary with keys:
            - "position": [x, y] top-left corner coordinates
            - "size": [w, h] width and height of bounding box

    Returns:
        torch.Tensor: Binary mask of shape [1, 1, H, W], dtype=float32
                     where 1 = background (attackable), 0 = foreground (protected)

    Example:
        >>> mask = bbox_mask((224, 224), {"position": [50, 50], "size": [100, 100]})
        >>> mask.shape
        torch.Size([1, 1, 224, 224])
        >>> mask[:, :, 50:150, 50:150].sum()  # Protected region
        tensor(0.)
        >>> mask[:, :, :50, :].sum()  # Background region
        tensor(11200.)  # 50 * 224 pixels
    """
    H, W = image_size
    x, y = position_dict["position"]
    w, h = position_dict["size"]

    # Create mask with all ones (entire image is attackable by default)
    mask = torch.ones(1, 1, H, W, dtype=torch.float32)

    # Zero out the foreground region (protect the object)
    # Ensure coordinates stay within image bounds
    x_end = min(x + w, W)
    y_end = min(y + h, H)
    x = max(0, x)
    y = max(0, y)

    mask[:, :, y:y_end, x:x_end] = 0

    return mask


def sam_mask(
    image: torch.Tensor,
    position_dict: Dict,
    sam_checkpoint: str = "sam_vit_h_4b8939.pth",
    device: str = "cuda"
) -> torch.Tensor:
    """
    Create binary mask using Segment Anything Model (SAM).

    This provides more precise foreground/background separation compared to bbox_mask
    by using SAM's state-of-the-art segmentation capabilities. The bounding box from
    position_dict is used as a prompt to SAM.

    Args:
        image: Input image tensor of shape [1, 3, H, W] or [3, H, W]
        position_dict: Dictionary with "position" and "size" keys (same as bbox_mask)
        sam_checkpoint: Path to SAM checkpoint file
        device: Device to run SAM on ('cuda' or 'cpu')

    Returns:
        torch.Tensor: Binary mask of shape [1, 1, H, W], dtype=float32
                     where 1 = background (attackable), 0 = foreground (protected)

    Note:
        Requires segment-anything to be installed:
        pip install git+https://github.com/facebookresearch/segment-anything.git

        SAM checkpoint can be downloaded from:
        https://github.com/facebookresearch/segment-anything#model-checkpoints
    """
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        raise ImportError(
            "segment-anything is required for sam_mask(). "
            "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )

    # Convert image to numpy for SAM
    if image.dim() == 4:
        image_np = image[0].cpu().numpy().transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
    else:
        image_np = image.cpu().numpy().transpose(1, 2, 0)

    # Denormalize if needed (SAM expects [0, 255] range)
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)

    # Initialize SAM
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_np)

    # Create bounding box prompt from position_dict
    x, y = position_dict["position"]
    w, h = position_dict["size"]
    bbox = np.array([x, y, x + w, y + h])

    # Generate mask using SAM
    masks, scores, logits = predictor.predict(
        box=bbox,
        multimask_output=False  # Single mask output
    )

    # SAM outputs foreground=1, we need background=1, so invert
    foreground_mask = masks[0]  # Shape: [H, W]
    background_mask = 1.0 - foreground_mask

    # Convert to torch tensor with correct shape [1, 1, H, W]
    mask = torch.from_numpy(background_mask).float().unsqueeze(0).unsqueeze(0)

    return mask


def saliency_mask(
    image: torch.Tensor,
    position_dict: Dict,
    threshold_multiplier: float = 0.5
) -> torch.Tensor:
    """
    Create binary mask using saliency detection.

    This uses OpenCV's spectral residual saliency detection to refine the
    bounding box mask. Pixels with high saliency within the bbox are marked
    as foreground, others as background.

    Args:
        image: Input image tensor of shape [1, 3, H, W] or [3, H, W]
        position_dict: Dictionary with "position" and "size" keys
        threshold_multiplier: Multiplier for adaptive thresholding
                            Higher values = more conservative (larger foreground)

    Returns:
        torch.Tensor: Binary mask of shape [1, 1, H, W], dtype=float32
                     where 1 = background (attackable), 0 = foreground (protected)

    Note:
        Requires opencv-python to be installed:
        pip install opencv-python
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for saliency_mask(). "
            "Install with: pip install opencv-python"
        )

    # Convert image to numpy
    if image.dim() == 4:
        image_np = image[0].cpu().numpy().transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
    else:
        image_np = image.cpu().numpy().transpose(1, 2, 0)

    # Convert to [0, 255] range if needed
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)

    # Compute saliency map
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency.computeSaliency(image_np)

    if not success:
        # Fall back to bbox mask if saliency computation fails
        H, W = image_np.shape[:2]
        return bbox_mask((H, W), position_dict)

    # Get bbox coordinates
    x, y = position_dict["position"]
    w, h = position_dict["size"]
    H, W = image_np.shape[:2]

    # Ensure coordinates are within bounds
    x_end = min(x + w, W)
    y_end = min(y + h, H)
    x = max(0, x)
    y = max(0, y)

    # Start with all background (ones)
    mask = np.ones((H, W), dtype=np.float32)

    # Within the bounding box, use saliency to determine foreground
    roi_saliency = saliency_map[y:y_end, x:x_end]

    if roi_saliency.size > 0:
        # Adaptive thresholding based on ROI statistics
        mean_saliency = np.mean(roi_saliency)
        std_saliency = np.std(roi_saliency)
        threshold = mean_saliency + threshold_multiplier * std_saliency

        # Mark high-saliency pixels as foreground (0)
        foreground_roi = (roi_saliency >= threshold).astype(np.float32)
        mask[y:y_end, x:x_end] = 1.0 - foreground_roi

    # Convert to torch tensor with shape [1, 1, H, W]
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)

    return mask_tensor


def get_segmentation_mask(
    image_size: Tuple[int, int],
    position_dict: Dict,
    method: str = 'bbox',
    image: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """
    Generate segmentation mask using specified method.

    This is the main interface for mask generation. It dispatches to the
    appropriate masking function based on the method parameter.

    Args:
        image_size: (H, W) tuple specifying image dimensions
        position_dict: Dictionary with "position" and "size" keys
        method: Masking method to use ('bbox', 'sam', or 'saliency')
        image: Input image tensor (required for 'sam' and 'saliency' methods)
        **kwargs: Additional arguments passed to the specific masking function

    Returns:
        torch.Tensor: Binary mask of shape [1, 1, H, W], dtype=float32
                     where 1 = background (attackable), 0 = foreground (protected)

    Raises:
        ValueError: If method is not recognized or required arguments are missing

    Examples:
        >>> # Bounding box mask (no image needed)
        >>> mask = get_segmentation_mask((224, 224), {"position": [50, 50], "size": [100, 100]})

        >>> # SAM mask (requires image)
        >>> mask = get_segmentation_mask(
        ...     (224, 224),
        ...     {"position": [50, 50], "size": [100, 100]},
        ...     method='sam',
        ...     image=image_tensor
        ... )

        >>> # Saliency mask (requires image)
        >>> mask = get_segmentation_mask(
        ...     (224, 224),
        ...     {"position": [50, 50], "size": [100, 100]},
        ...     method='saliency',
        ...     image=image_tensor
        ... )
    """
    if method == 'bbox':
        return bbox_mask(image_size, position_dict)

    elif method == 'sam':
        if image is None:
            raise ValueError("image argument is required for 'sam' method")
        return sam_mask(image, position_dict, **kwargs)

    elif method == 'saliency':
        if image is None:
            raise ValueError("image argument is required for 'saliency' method")
        return saliency_mask(image, position_dict, **kwargs)

    else:
        raise ValueError(
            f"Unknown segmentation method: {method}. "
            f"Supported methods are: 'bbox', 'sam', 'saliency'"
        )


def apply_mask_erosion(
    mask: torch.Tensor,
    erosion_pixels: int = 5
) -> torch.Tensor:
    """
    Apply erosion to mask to create a safety margin around protected regions.

    This expands the protected (foreground) region by eroding the background,
    creating a buffer zone to prevent perturbations from leaking onto object edges.

    Args:
        mask: Binary mask of shape [1, 1, H, W]
        erosion_pixels: Number of pixels to erode (expand foreground by this amount)

    Returns:
        torch.Tensor: Eroded mask of same shape
    """
    import torch.nn.functional as F

    # Invert mask so foreground is 1
    inverted = 1.0 - mask

    # Create erosion kernel
    kernel_size = 2 * erosion_pixels + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)

    # Apply max pooling to erode (expand foreground)
    padding = erosion_pixels
    eroded = F.max_pool2d(inverted, kernel_size, stride=1, padding=padding)

    # Invert back so background is 1
    result = 1.0 - eroded

    return result


def apply_mask_feathering(
    mask: torch.Tensor,
    sigma: float = 2.0
) -> torch.Tensor:
    """
    Apply Gaussian blur to mask edges for smooth transitions.

    This creates feathered boundaries between protected and attackable regions,
    reducing visible artifacts at mask boundaries.

    Args:
        mask: Binary mask of shape [1, 1, H, W]
        sigma: Gaussian blur sigma (higher = more blur)

    Returns:
        torch.Tensor: Feathered mask of same shape
    """
    import torch.nn.functional as F

    # Create Gaussian kernel
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()

    # Create 2D kernel
    kernel_1d = gauss.unsqueeze(1)
    kernel_2d = kernel_1d @ kernel_1d.t()
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).to(mask.device)

    # Apply Gaussian blur
    padding = kernel_size // 2
    feathered = F.conv2d(mask, kernel_2d, padding=padding)

    return feathered
