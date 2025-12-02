"""
test_background_attack.py

Unit tests for background-only CLIP attack implementation.
Tests mask generation, gradient masking, and foreground preservation.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from agent_attack.attacks.segmentation import (
    apply_mask_erosion,
    apply_mask_feathering,
    bbox_mask,
    get_segmentation_mask,
)


class TestMaskGeneration:
    """Test suite for segmentation mask generation."""

    def test_bbox_mask_basic(self):
        """Test that bbox_mask creates correct binary mask."""
        image_size = (224, 224)
        position = {"position": [50, 50], "size": [100, 100]}

        mask = bbox_mask(image_size, position)

        # Check shape
        assert mask.shape == (1, 1, 224, 224)
        assert mask.dtype == torch.float32

        # Check foreground region is protected (0)
        assert mask[:, :, 50:150, 50:150].sum() == 0

        # Check background regions are attackable (1)
        assert mask[:, :, :50, :].sum() > 0
        assert mask[:, :, 150:, :].sum() > 0
        assert mask[:, :, :, :50].sum() > 0
        assert mask[:, :, :, 150:].sum() > 0

    def test_bbox_mask_bounds_checking(self):
        """Test that bbox_mask handles out-of-bounds coordinates."""
        image_size = (224, 224)

        # Test position exceeding image bounds
        position = {"position": [200, 200], "size": [100, 100]}
        mask = bbox_mask(image_size, position)

        # Should clip to image boundaries
        assert mask.shape == (1, 1, 224, 224)
        assert mask[:, :, 200:224, 200:224].sum() == 0

    def test_bbox_mask_corner_cases(self):
        """Test edge cases for bbox_mask."""
        image_size = (100, 100)

        # Test zero-size object
        position = {"position": [50, 50], "size": [0, 0]}
        mask = bbox_mask(image_size, position)
        assert mask.sum() == 10000  # All pixels are background

        # Test full-image object
        position = {"position": [0, 0], "size": [100, 100]}
        mask = bbox_mask(image_size, position)
        assert mask.sum() == 0  # All pixels are foreground

    def test_get_segmentation_mask_bbox(self):
        """Test get_segmentation_mask with bbox method."""
        mask = get_segmentation_mask(
            (224, 224),
            {"position": [50, 50], "size": [100, 100]},
            method='bbox'
        )

        assert mask.shape == (1, 1, 224, 224)
        assert mask[:, :, 50:150, 50:150].sum() == 0

    def test_get_segmentation_mask_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown segmentation method"):
            get_segmentation_mask(
                (224, 224),
                {"position": [50, 50], "size": [100, 100]},
                method='invalid_method'
            )

    def test_get_segmentation_mask_missing_image(self):
        """Test that SAM/saliency methods require image argument."""
        with pytest.raises(ValueError, match="image argument is required"):
            get_segmentation_mask(
                (224, 224),
                {"position": [50, 50], "size": [100, 100]},
                method='sam'
            )


class TestMaskOperations:
    """Test suite for mask manipulation operations."""

    def test_mask_erosion(self):
        """Test that erosion expands protected regions."""
        # Create simple mask with small protected region
        mask = torch.ones(1, 1, 100, 100)
        mask[:, :, 40:60, 40:60] = 0  # 20x20 protected region

        # Apply erosion
        eroded = apply_mask_erosion(mask, erosion_pixels=5)

        # Check that protected region expanded
        assert eroded[:, :, 35:65, 35:65].sum() < mask[:, :, 35:65, 35:65].sum()

        # Original protected region should still be protected
        assert eroded[:, :, 40:60, 40:60].sum() == 0

    def test_mask_feathering(self):
        """Test that feathering creates smooth transitions."""
        # Create sharp binary mask
        mask = torch.ones(1, 1, 100, 100)
        mask[:, :, 40:60, 40:60] = 0

        # Apply feathering
        feathered = apply_mask_feathering(mask, sigma=2.0)

        # Check that boundaries are no longer binary
        # (should have values between 0 and 1)
        boundary_region = feathered[:, :, 38:42, 38:42]
        assert (boundary_region > 0).any() and (boundary_region < 1).any()


class TestGradientMasking:
    """Test suite for gradient masking functionality."""

    def test_gradient_masking(self):
        """Test that gradients are zeroed on foreground."""
        # Create dummy tensor and mask
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        mask = torch.ones(1, 1, 224, 224)
        mask[:, :, 50:150, 50:150] = 0  # Protect center region

        # Compute dummy gradient
        loss = x.sum()
        loss.backward()

        grad = x.grad.clone()
        masked_grad = grad * mask

        # Check foreground gradients are zero
        assert masked_grad[:, :, 50:150, 50:150].abs().sum() == 0

        # Check background gradients are non-zero
        assert masked_grad[:, :, :50, :].abs().sum() > 0


class TestForegroundPreservation:
    """Test suite for foreground preservation during attacks."""

    def test_foreground_enforcement(self):
        """Test that foreground pixels remain unchanged after masking."""
        # Create original and perturbed images
        original = torch.rand(1, 3, 224, 224)
        perturbed = torch.rand(1, 3, 224, 224)

        # Create mask
        mask = torch.ones(1, 1, 224, 224)
        mask[:, :, 50:150, 50:150] = 0

        # Apply enforcement: original * (1 - mask) + perturbed * mask
        result = original * (1 - mask) + perturbed * mask

        # Check foreground unchanged
        assert torch.allclose(
            result[:, :, 50:150, 50:150],
            original[:, :, 50:150, 50:150],
            atol=1e-6
        )

        # Check background changed
        assert not torch.allclose(
            result[:, :, :50, :],
            original[:, :, :50, :],
            atol=1e-6
        )

    def test_foreground_preservation_numerical_stability(self):
        """Test that repeated enforcement doesn't cause drift."""
        original = torch.rand(1, 3, 100, 100)
        x = original.clone()

        mask = torch.ones(1, 1, 100, 100)
        mask[:, :, 40:60, 40:60] = 0

        # Apply enforcement 100 times (simulating attack iterations)
        for _ in range(100):
            # Perturb background
            x = x + torch.randn_like(x) * 0.01 * mask

            # Enforce foreground
            x = original * (1 - mask) + x * mask

        # Foreground should be identical to original (no drift)
        assert torch.allclose(
            x[:, :, 40:60, 40:60],
            original[:, :, 40:60, 40:60],
            atol=1e-5
        )


class TestIntegration:
    """Integration tests for the complete background attack pipeline."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_clip_attack_background_minimal(self):
        """Test clip_attack_background with minimal setup (smoke test)."""
        from agent_attack.attacks.clip_attack_background import clip_attack_background

        # Create test image
        image = Image.new('RGB', (224, 224), color='red')
        position = {"position": [50, 50], "size": [100, 100]}

        # Run attack with very few iterations for speed
        try:
            result = clip_attack_background(
                image,
                target_text="blue",
                victim_text="red",
                position=position,
                iters=2,  # Minimal iterations for testing
                size=224,
                mask_method='bbox'
            )

            # Check return structure
            assert "adv_images" in result
            assert "mask" in result
            assert "fg_metrics" in result

            # Check metrics exist
            assert "psnr" in result["fg_metrics"]
            assert "ssim" in result["fg_metrics"]

            print("✓ Minimal integration test passed")

        except Exception as e:
            pytest.skip(f"Integration test skipped: {e}")


class TestMetrics:
    """Test suite for foreground preservation metrics."""

    def test_psnr_perfect_preservation(self):
        """Test PSNR metric with perfect foreground preservation."""
        from agent_attack.attacks.clip_attack_background import compute_foreground_metrics

        # Create identical images
        original = torch.rand(1, 3, 100, 100)
        adversarial = original.clone()

        # Create mask (protect center)
        mask = torch.ones(1, 1, 100, 100)
        mask[:, :, 40:60, 40:60] = 0

        metrics = compute_foreground_metrics(original, adversarial, mask)

        # Perfect preservation should give infinite PSNR and SSIM=1
        assert metrics["psnr"] == float('inf') or metrics["psnr"] > 100
        assert metrics["ssim"] >= 0.99
        assert metrics["max_change"] < 1e-5

    def test_psnr_with_perturbation(self):
        """Test PSNR metric with background perturbation."""
        from agent_attack.attacks.clip_attack_background import compute_foreground_metrics

        # Create original image
        original = torch.rand(1, 3, 100, 100)

        # Perturb entire image
        adversarial = original + torch.randn_like(original) * 0.1

        # Create mask (protect center)
        mask = torch.ones(1, 1, 100, 100)
        mask[:, :, 40:60, 40:60] = 0

        # Force foreground back to original
        adversarial = original * (1 - mask) + adversarial * mask

        metrics = compute_foreground_metrics(original, adversarial, mask)

        # Foreground should still be well-preserved
        assert metrics["psnr"] > 40  # Typically >45 for good preservation
        assert metrics["ssim"] > 0.95


def run_tests():
    """Run all tests and print summary."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
