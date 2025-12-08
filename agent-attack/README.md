# VLM Surrogate Attacks: Background-Focused Adversarial Examples

## The Problem Being Solved

Vision-Language Models like GPT-4V, Claude, and LLaVA use a component called CLIP to understand images. CLIP was trained on billions of image-caption pairs from the internet, learning to connect visual features with text descriptions. When someone uploads an image of a product to an AI shopping assistant, CLIP helps the model "see" what's in the image.

Here's the vulnerability this work exploits. CLIP has a documented attention bias toward backgrounds. When CLIP looks at a photo of a watch on a table, it doesn't just analyze the watch. It analyzes the table, the lighting, the shadows, everything. Researchers have shown that CLIP often pays disproportionate attention to contextual background features rather than the central subject.

This means it's possible to manipulate what CLIP "understands" about an image by changing only the background pixels while leaving the actual product completely untouched. A human looking at two photos side by side would see an identical watch. But CLIP might interpret one as "luxury item" and the other as "discount product" based solely on imperceptible background noise.

This matters for AI agent security. Imagine an AI shopping agent that uses vision models to compare products. An attacker could post product images with adversarially crafted backgrounds that make expensive items appear cheap to the AI while looking completely normal to human moderators reviewing the listings.

---

## What Adversarial Attacks Actually Are

For a non-technical understanding, imagine a photo of a cat. To a human, it's obviously a cat. But if specific, carefully calculated noise is added to certain pixels, an AI classifier can be made to think it's seeing a toaster. The changes are so small that humans can't perceive them, but they push the AI's mathematical interpretation across a decision boundary.

The technical mechanism works like this. Neural networks learn to map inputs (images) to outputs (classifications or embeddings) through millions of learned parameters. These mappings are continuous and differentiable, meaning small input changes produce small output changes in a smooth, calculable way. Adversarial attacks exploit this by computing the gradient of the output with respect to the input. The gradient tells exactly which direction to nudge each pixel to maximally change the output.

Standard attacks like PGD (Projected Gradient Descent) perturb all pixels. This work constrains perturbations to background regions only.

---

## The Architecture of the Solution

This commit creates three new files forming a modular attack infrastructure.

### `segmentation.py`
Handles mask generation. It answers the question "which pixels are foreground (protected) and which are background (attackable)?" Three masking strategies are implemented:

*   **bbox**: The MVP approach. Given a bounding box around the object, everything inside the box is foreground, everything outside is background. This is fast, requires no additional models, and works reliably. If the product is at position (100,100) with size (200,200), a binary mask is created where that rectangular region is zeros (protected) and everywhere else is ones (attackable).
*   **SAM**: Uses Meta's Segment Anything Model for precise object boundaries. Instead of a crude rectangle, SAM can trace the exact silhouette of a watch, excluding shadows and reflections that extend beyond the bounding box. This costs more compute but produces tighter foreground isolation.
*   **saliency**: Uses gradient-based attention maps to identify what the neural network itself considers "important" pixels. This is theoretically elegant because it protects exactly what the model focuses on, but it's computationally expensive and requires additional forward passes.

### `clip_attack_background.py`
The core attack implementation. It extends an existing attack class called `SSA_CommonWeakness` with spatial masking capabilities.

The original `SSA_CommonWeakness` attack works like this. It uses an ensemble of four CLIP models with different architectures (ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14-336). The ensemble matters because if perturbations are optimized against only one model, they might not transfer to other models. By attacking multiple CLIP variants simultaneously, perturbations are found that exploit shared vulnerabilities across the CLIP family.

The attack optimizes a loss function that pushes the image embedding toward a target text ("this is the cheapest product") and optionally away from a victim text ("this is an expensive product"). Over hundreds of iterations, the attack computes gradients and updates pixels to minimize this loss while staying within an epsilon-ball constraint (typically 16/255 per pixel, imperceptible to humans).

The `SSA_CommonWeakness_Masked` subclass adds four-point mask enforcement, which is the critical innovation.

---

## Four-Point Mask Enforcement

This is the technical core of the work. Simply multiplying gradients by a mask isn't sufficient to guarantee foreground preservation. Numerical errors, momentum accumulation, and clamping operations can leak perturbations into protected regions. The mask is enforced at four distinct points in the optimization loop.

1.  **During random initialization**: Many attacks start with small random noise to escape local minima. After adding this noise, the original foreground is immediately composited back onto the image.
    `x = original_x * (1 - mask) + x * mask`
    This means "take original pixels where mask is 0 (foreground), take perturbed pixels where mask is 1 (background)."

2.  **After gradient computation**: Before using gradients to update the image, gradients on foreground pixels are zeroed out with `grad = grad * mask`. This prevents any gradient signal from flowing into protected regions.

3.  **After the inner loop update**: Even with masked gradients, momentum terms can accumulate information that slightly affects protected pixels through numerical precision issues. Foreground pixels are forced back to their original values after each update step.

4.  **After the `end_attack` method**: The parent class `SSA_CommonWeakness` has its own `end_attack` processing (outer momentum updates, spectrum transformations). These operations might introduce tiny foreground changes. One final enforcement is applied after this processing completes.

This belt-and-suspenders approach ensures that regardless of what happens inside the optimization loop, the foreground emerges mathematically identical to the input.

---

## The Optimization Loop Structure

Each iteration of the attack follows this sequence:

1.  **`begin_attack`** is called which resets internal state and prepares the gradient recording infrastructure.
2.  The loop runs through all four **CLIP models**. For each model, the loss between the current image embedding and the target text embedding is computed. The loss function uses cosine similarity in CLIP's joint embedding space. Backpropagation gets gradients with respect to input pixels, the mask is applied to zero foreground gradients, and the gradient is recorded for later aggregation.
3.  After collecting gradients from all models, the image is updated using **momentum-based gradient descent**. The inner momentum helps escape local minima and stabilizes updates across iterations. Pixel values are clamped to [0,1] and to the epsilon-ball around the original image.
4.  **`end_attack`** handles outer loop processing, including spectrum-based augmentations that improve transferability. SSA (Spectrum Simulation Attack) transforms gradients into frequency space using DCT, applies random filtering, and transforms back. This prevents overfitting to specific frequency patterns of the surrogate models.
5.  Every 100 iterations, a **snapshot** of the adversarial image is saved. This allows examination of attack progression and choosing early-stopped versions if the final iteration overfits.

---

## Foreground Preservation Metrics

The `compute_foreground_metrics` function quantifies how well the mask enforcement worked.

*   **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-level fidelity. It compares original and adversarial foreground pixels. Perfect preservation yields infinite PSNR. Values above 40dB indicate negligible changes invisible to humans.
*   **SSIM (Structural Similarity Index)**: Measures perceptual similarity accounting for luminance, contrast, and structure. It's more aligned with human vision than raw pixel comparisons. A score of 1.0 means identical images.
*   **Max change**: Reports the largest absolute pixel difference in the foreground. With proper mask enforcement, this should be exactly 0.0 (or floating-point epsilon). Any non-zero value indicates mask leakage.

---

## The Broader Research Context

This work connects to several important research threads.

*   **Surrogate attacks exploit transferability.** Gradients can't be computed through proprietary models like GPT-4V because there's no access to their weights. But perturbations optimized against open-source surrogates (CLIP models under control) often transfer to black-box targets. This is because different vision models learn similar low-level features and share similar vulnerability surfaces.
*   **Background attacks are a specific form of constrained adversarial examples.** Prior work on patch attacks showed classifiers can be fooled by modifying small image regions. This approach inverts that by protecting small regions and attacking everything else.
*   **The CLIP background attention bias** was documented in papers analyzing CLIP's failure modes. Models trained on internet-scraped data inherit biases from how photographers compose images. CLIP learns spurious correlations between backgrounds and concepts because training data isn't controlled for confounders.

---

## What the Test File Validates

The `test_background_attack.py` file provides verification that mask enforcement works correctly across edge cases.

*   Tests verify that foreground pixels remain mathematically unchanged (`max_change == 0`).
*   Mask generation is verified to produce correct shapes and semantics.
*   The attack is confirmed to produce valid adversarial images that satisfy epsilon constraints.
*   Tests check that the attack runs on both CPU and CUDA without errors.

---

## Practical Implications

This infrastructure enables systematic study of a real vulnerability in deployed AI systems. Shopping agents, content moderation systems, and visual search engines all rely on vision encoders descended from CLIP-like architectures.

An attacker with this code could upload product images where the item looks identical to humans but appears misclassified to AI agents. A legitimate security researcher could use this to benchmark model robustness before deployment.

The four-point enforcement pattern is a contribution to adversarial ML methodology. Prior masked attack implementations often leaked perturbations through numerical edge cases. The explicit multi-point enforcement provides stronger guarantees.

The modular design separating segmentation, attack logic, and metrics allows swapping components independently. SAM-based masks can be tested against bbox masks without rewriting attack code. New CLIP variants can be added to the ensemble without touching segmentation.

---

*This is Phase 1 infrastructure. The next phases would involve empirical validation against real VLM targets, analysis of transferability rates, and potentially defense mechanisms that detect background-only perturbations.*
