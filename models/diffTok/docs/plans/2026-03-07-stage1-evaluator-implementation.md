# Stage 1 Evaluator Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign Stage 1 evaluator to support three visualization modes (noise recovery, timestep sweep, attention mask sweep) with config-driven enable/disable and clean visualizer interface.

**Architecture:** Separate visualization logic from evaluation logic. Three independent evaluation functions, each callable standalone. Config-driven enable/disable via YAML. Minimal metrics, visualization-focused.

**Tech Stack:** PyTorch, Matplotlib, YAML configs

---

## Task 1: Update Configuration Structure

**Files:**
- Modify: `configs/stage1/base.yaml:272-287`

**Step 1: Add visualization configuration to base.yaml**

Replace the existing `evaluation` section with:

```yaml
# =============================================================================
# Evaluation Configuration
# =============================================================================
evaluation:
  # Evaluate every N steps
  eval_every_steps: 10000

  # =============================================================================
  # Metrics Configuration (computed on entire validation set)
  # =============================================================================
  metrics:
    enabled: true
    types:
      - mse    # Mean Squared Error
      - psnr   # Peak Signal-to-Noise Ratio
      - fid    # Fréchet Inception Distance

  # =============================================================================
  # Visualization Configuration (only on num_vis_samples images)
  # =============================================================================
  visualization:
    # Evaluation type 1: Noise recovery (n-step reconstruction from pure noise)
    noise_recovery:
      enabled: true
      num_steps: 4  # Number of forward passes for reconstruction
      num_vis_samples: 8  # Number of images to display vertically

    # Evaluation type 2: Timestep sweep (single reconstruction at different timesteps)
    timestep_sweep:
      enabled: true
      initial_timesteps: [0.25, 0.5, 0.75, 1.0]
      num_vis_samples: 8  # Number of images to display vertically

    # Evaluation type 3: Attention mask sweep (varying masked token counts)
    attn_mask_sweep:
      enabled: true
      mask_token_nums: [0, 32, 64, 96, 128, 160, 192]
      num_vis_samples: 8  # Number of images to display vertically
```

**Step 2: Verify YAML syntax**

```bash
python -c "import yaml; yaml.safe_load(open('configs/stage1/base.yaml'))"
```

**Step 3: Commit**

```bash
git add configs/stage1/base.yaml
git commit -m "feat(evaluation): add visualization configuration structure"
```

---

## Task 2: Create FID Metric Helper

**Files:**
- Modify: `src/evaluators/tools/metrics.py`

**Step 1: Implement compute_fid function**

```python
def compute_fid(real_images, generated_images, device='cuda'):
    """
    Compute Fréchet Inception Distance between real and generated images.

    Args:
        real_images: Tensor of real images [N, C, H, W]
        generated_images: Tensor of generated images [N, C, H, W]
        device: Device to run computation on

    Returns:
        fid_score: FID score (lower is better)
    """
    import torch
    import numpy as np
    from torchvision import models
    from torchvision.models import Inception_V3_Weights
    from scipy import linalg

    # Load pretrained Inception v3 model
    inception = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    inception.to(device)
    inception.eval()

    # Resize images to 299x299 for Inception v3
    from torchvision import transforms
    resize = transforms.Resize((299, 299))

    with torch.no_grad():
        # Get features for real images
        real_resized = torch.stack([resize(img) for img in real_images])
        real_pred = inception(real_resized.to(device)).cpu()

        # Get features for generated images
        gen_resized = torch.stack([resize(img) for img in generated_images])
        gen_pred = inception(gen_resized.to(device)).cpu()

    # Compute mean and covariance
    mu_real = np.mean(real_pred.numpy(), axis=0)
    sigma_real = np.cov(real_pred.numpy(), rowvar=False)

    mu_gen = np.mean(gen_pred.numpy(), axis=0)
    sigma_gen = np.cov(gen_pred.numpy(), rowvar=False)

    # Compute FID
    diff = mu_real - mu_gen
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)

    return float(fid_score)
```

**Step 2: Commit**

```bash
git add src/evaluators/tools/metrics.py
git commit -m "feat(metrics): add FID computation function"
```

---

## Task 3: Create Visualizer Helper Functions

**Files:**
- Modify: `src/evaluators/tools/visualizers.py`

**Step 1: Implement concatenate_with_arrow function**

Add to `src/evaluators/tools/visualizers.py` (after the existing `save_reconstruction_comparison` function):

```python
def concatenate_with_arrow(input_img, output_img, gap_size=8):
    """
    Concatenate two images horizontally with an arrow in between.

    Creates: [input_img] -> [output_img]
    Arrow is drawn in the gap between images.

    Args:
        input_img: Input image tensor [C, H, W]
        output_img: Output image tensor [C, H, W]
        gap_size: Size of gap between images (default: 8)

    Returns:
        Concatenated image tensor [C, H, W_total]
    """
    import torch

    C, H, W = input_img.shape

    # Calculate total width: img1 + gap + img2
    total_width = W + gap_size + W

    # Create output tensor
    result = torch.zeros(C, H, total_width)

    # Place input image on the left
    result[:, :, :W] = input_img

    # Place output image on the right
    result[:, :, W + gap_size:] = output_img

    # Draw arrow in the gap (simple horizontal line)
    gap_start = W
    gap_end = W + gap_size
    center_y = H // 2

    # Draw arrow line (white on all channels)
    for x in range(gap_start, gap_end):
        result[:, center_y, x] = 1.0

    # Draw arrow head
    result[:, center_y - 2:center_y + 3, gap_end - 2:gap_end] = 1.0

    return result
```

**Step 2: Implement create_grid_visualizer function**

Add to `src/evaluators/tools/visualizers.py`:

```python
def create_grid_visualizer(
    image_name,
    xlabel,
    ylabel,
    xticks,
    yticks,
    images_2d,
    save_path,
    project_name=None
):
    """
    Create a flexible grid visualization for evaluation results.

    Args:
        image_name: Name/title for the image (used in logging)
        xlabel: X-axis label (empty string to hide)
        ylabel: Y-axis label (empty string to hide)
        xticks: List of x-axis tick labels
        yticks: List of y-axis tick labels (can be empty)
        images_2d: 2D list of image tensors [row][col], each [C, H, W]
        save_path: Path to save the visualization
        project_name: Optional project name for main title
    """
    import matplotlib.pyplot as plt
    import os

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    num_rows = len(images_2d)
    num_cols = len(images_2d[0]) if num_rows > 0 else 0

    # Create figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.5, num_rows * 2.5))

    # Handle single row/column case
    if num_rows == 1 and num_cols == 1:
        axes = [[axes]]
    elif num_rows == 1:
        axes = [axes]
    elif num_cols == 1:
        axes = [[ax] for ax in axes]

    # Plot images
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axes[i][j]
            img = images_2d[i][j]

            # Convert tensor to numpy
            if isinstance(img, torch.Tensor):
                img_np = img.detach().cpu().permute(1, 2, 0).numpy()
            else:
                img_np = img

            # Clip to [0, 1]
            img_np = img_np.clip(0, 1)

            ax.imshow(img_np)

            # Set x-axis labels (only on first row)
            if i == 0 and j < len(xticks):
                ax.set_title(xticks[j], fontsize=10, fontweight='bold')

            # Set y-axis labels (only on first column)
            if j == 0 and i < len(yticks):
                ax.set_ylabel(yticks[i], fontsize=10)

            ax.axis('off')

    # Set axis labels
    if xlabel:
        fig.supxlabel(xlabel, fontsize=12)
    if ylabel:
        fig.supylabel(ylabel, fontsize=12)

    # Add project name as main title
    if project_name:
        fig.suptitle(project_name, fontsize=14, fontweight='bold', y=0.95)

    # Adjust layout
    plt.tight_layout()
    if project_name:
        plt.subplots_adjust(top=0.90)

    # Save figure
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)

    print(f"✅ Saved {image_name} visualization to {save_path}")
```

**Step 3: Commit**

```bash
git add src/evaluators/tools/visualizers.py
git commit -m "feat(visualizer): add concatenate_with_arrow and create_grid_visualizer"
```

---

## Task 3: Implement Noise Recovery Evaluation

**Files:**
- Modify: `src/evaluators/stage1_evaluator.py`

**Step 1: Implement visualize_noise_recovery function**

Add to `src/evaluators/stage1_evaluator.py` (after the `compute_codebook_stats` function):

```python
def visualize_noise_recovery(
    model, accelerator, dataloader, noise_scheduler,
    config, save_dir, epoch, project_name=None
):
    """
    Generate n-step reconstruction visualization from pure noise.

    Process:
    1. Select N images from first batch only
    2. For each image:
       - Store original
       - Tokenize to get latent tokens
       - Generate pure noise image
       - Run n denoise passes, saving each output
    3. Build grid with visualizer

    Args:
        model: Poet model
        accelerator: Accelerator instance
        dataloader: Validation data loader
        noise_scheduler: Noise scheduler
        config: Dict with 'num_steps' and 'num_vis_samples'
        save_dir: Directory to save visualizations
        epoch: Current epoch
        project_name: Optional project name

    Returns:
        None (saves visualization to disk)
    """
    from tools.visualizers import create_grid_visualizer
    import os

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    num_steps = config.get('num_steps', 4)
    num_vis_samples = config.get('num_vis_samples', 8)

    images_2d = []

    with torch.no_grad():
        # Only process first batch for visualization
        images, labels = next(iter(dataloader))
        images = images.to(accelerator.device)
        labels = labels.to(accelerator.device)

        for img_idx in range(min(num_vis_samples, images.shape[0])):
            single_image = images[img_idx:img_idx+1]
            single_label = labels[img_idx:img_idx+1]

            # Tokenize to get latent tokens
            cls_token, latent_tokens, _ = model.tokenize(single_image, single_label)

            # Store row for this image
            row_images = []

            # Add original image
            row_images.append(single_image[0])

            # Generate pure noise image
            noise_image = torch.randn_like(single_image)
            row_images.append(noise_image[0])

            # Iterative reconstruction
            current_image = noise_image
            for step in range(num_steps):
                # Denoise pass with decreasing timestep
                t_value = 1.0 - (step + 1) / (num_steps + 1)
                t_tensor = torch.full((1,), t_value, device=accelerator.device)
                output_image = model.denoise(
                    current_image, cls_token, latent_tokens,
                    t_tensor, num_masked_tokens=None
                )

                # Add to row
                row_images.append(output_image[0])

                # Use output as next input
                current_image = output_image

            images_2d.append(row_images)

    # Create visualization
    xticks = ["original", "noise"] + [f"step{i+1}" for i in range(num_steps)]
    save_path = os.path.join(save_dir, f'noise_recovery_epoch{epoch:04d}.png')

    create_grid_visualizer(
        image_name="noise_recovery",
        xlabel="",
        ylabel="",
        xticks=xticks,
        yticks=[],
        images_2d=images_2d,
        save_path=save_path,
        project_name=project_name
    )
```

**Step 2: Implement calculate_noise_recovery_metrics function**

Add to `src/evaluators/stage1_evaluator.py`:

```python
def calculate_noise_recovery_metrics(
    model, accelerator, dataloader, noise_scheduler,
    config
):
    """
    Calculate metrics for n-step reconstruction from pure noise.

    Process:
    1. Iterate over ALL validation data
    2. For each image:
       - Tokenize to get latent tokens
       - Generate pure noise image
       - Run n denoise passes
       - Collect real and generated images for all metrics
    3. Compute all metrics from collected data

    Args:
        model: Poet model
        accelerator: Accelerator instance
        dataloader: Validation data loader
        noise_scheduler: Noise scheduler
        config: Dict with 'num_steps' and 'metrics' list

    Returns:
        Dict with average metrics (mse_per_step, psnr_per_step, fid_per_step)
    """
    import torch.nn.functional as F

    model.eval()

    num_steps = config.get('num_steps', 4)
    metrics_list = config.get('metrics', ['mse', 'psnr'])

    # Collect real and generated images for ALL metrics
    all_real_images = []
    all_generated_images = {step: [] for step in range(num_steps)}

    with torch.no_grad():
        # Process ALL validation data - only collect images
        for images, labels in dataloader:
            images = images.to(accelerator.device)
            labels = labels.to(accelerator.device)

            for img_idx in range(images.shape[0]):
                single_image = images[img_idx:img_idx+1]
                single_label = labels[img_idx:img_idx+1]

                # Tokenize to get latent tokens
                cls_token, latent_tokens, _ = model.tokenize(single_image, single_label)

                # Store original image
                all_real_images.append(single_image[0])

                # Generate pure noise image
                noise_image = torch.randn_like(single_image)

                # Iterative reconstruction
                current_image = noise_image
                for step in range(num_steps):
                    # Denoise pass with decreasing timestep
                    t_value = 1.0 - (step + 1) / (num_steps + 1)
                    t_tensor = torch.full((1,), t_value, device=accelerator.device)
                    output_image = model.denoise(
                        current_image, cls_token, latent_tokens,
                        t_tensor, num_masked_tokens=None
                    )

                    # Store generated image
                    all_generated_images[step].append(output_image[0])

                    # Use output as next input
                    current_image = output_image

    # Compute all metrics from collected data
    # Stack real images once (shared across all steps)
    real_batch = torch.stack(all_real_images)

    # Initialize metrics dict
    avg_metrics = {metric: [] for metric in metrics_list}

    for step in range(num_steps):
        gen_batch = torch.stack(all_generated_images[step])

        for metric in metrics_list:
            if metric == 'mse':
                mse = F.mse_loss(gen_batch, real_batch).item()
                avg_metrics[metric].append(mse)
            elif metric == 'psnr':
                mse = F.mse_loss(gen_batch, real_batch).item()
                psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse)))
                avg_metrics[metric].append(psnr)
            elif metric == 'fid':
                from tools.metrics import compute_fid
                fid_score = compute_fid(real_batch, gen_batch)
                avg_metrics[metric].append(fid_score)

    return avg_metrics
```

**Step 3: Commit**

```bash
git add src/evaluators/stage1_evaluator.py
git commit -m "feat(evaluator): implement noise recovery evaluation (separated viz and metrics)"
```

---

## Task 4: Implement Timestep Sweep Evaluation

**Files:**
- Modify: `src/evaluators/stage1_evaluator.py`

**Step 1: Implement visualize_timestep_sweep function**

Add to `src/evaluators/stage1_evaluator.py`:

```python
def visualize_timestep_sweep(
    model, accelerator, dataloader, noise_scheduler,
    config, save_dir, epoch, project_name=None
):
    """
    Generate visualization for single reconstruction at different timesteps.

    Process:
    1. Select N images from first batch
    2. For each timestep:
       - Tokenize to get latent tokens
       - Add noise at that level
       - Run single denoise pass
       - Concatenate input->output with arrow
    3. Build grid with visualizer

    Args:
        model: Poet model
        accelerator: Accelerator instance
        dataloader: Validation data loader
        noise_scheduler: Noise scheduler
        config: Dict with 'initial_timesteps' and 'num_vis_samples'
        save_dir: Directory to save visualizations
        epoch: Current epoch
        project_name: Optional project name

    Returns:
        None (saves visualization to disk)
    """
    from tools.visualizers import create_grid_visualizer, concatenate_with_arrow
    import os

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    timesteps = config.get('initial_timesteps', [0.25, 0.5, 0.75, 1.0])
    num_vis_samples = config.get('num_vis_samples', 8)

    images_2d = []

    with torch.no_grad():
        # Only process first batch for visualization
        images, labels = next(iter(dataloader))
        images = images.to(accelerator.device)
        labels = labels.to(accelerator.device)

        for img_idx in range(min(num_vis_samples, images.shape[0])):
            single_image = images[img_idx:img_idx+1]
            single_label = labels[img_idx:img_idx+1]

            # Tokenize to get latent tokens
            cls_token, latent_tokens, _ = model.tokenize(single_image, single_label)

            # Row for this image
            row_images = []

            # Column 0: Original image (no arrow)
            row_images.append(single_image[0])

            # For each timestep, create noisy->reconstructed pair
            for timestep in timesteps:
                # Add noise to image
                t_tensor = torch.full((1,), timestep, device=accelerator.device)
                noise = torch.randn_like(single_image)
                noisy_image = noise_scheduler.add_noise(single_image, noise, t_tensor)

                # Denoise pass
                output_image = model.denoise(
                    noisy_image, cls_token, latent_tokens,
                    t_tensor, num_masked_tokens=None
                )

                # Concatenate noisy -> output with arrow
                paired_image = concatenate_with_arrow(noisy_image[0], output_image[0])
                row_images.append(paired_image)

            images_2d.append(row_images)

    # Create visualization
    xticks = ["original"] + [str(t) for t in timesteps]
    save_path = os.path.join(save_dir, f'timestep_sweep_epoch{epoch:04d}.png')

    create_grid_visualizer(
        image_name="timestep_sweep",
        xlabel="",
        ylabel="",
        xticks=xticks,
        yticks=[],
        images_2d=images_2d,
        save_path=save_path,
        project_name=project_name
    )
```

**Step 2: Implement calculate_timestep_sweep_metrics function**

Add to `src/evaluators/stage1_evaluator.py`:

```python
def calculate_timestep_sweep_metrics(
    model, accelerator, dataloader, noise_scheduler,
    config
):
    """
    Calculate metrics for single reconstruction at different timesteps.

    Process:
    1. Iterate over ALL validation data
    2. For each timestep:
       - Tokenize to get latent tokens
       - Add noise at that level
       - Run single denoise pass
       - Collect real and generated images
    3. Compute all metrics from collected data

    Args:
        model: Poet model
        accelerator: Accelerator instance
        dataloader: Validation data loader
        noise_scheduler: Noise scheduler
        config: Dict with 'initial_timesteps' and 'metrics' list

    Returns:
        Dict with metrics per timestep
    """
    import torch.nn.functional as F

    model.eval()

    timesteps = config.get('initial_timesteps', [0.25, 0.5, 0.75, 1.0])
    metrics_list = config.get('metrics', ['mse', 'psnr'])

    # Collect real and generated images for ALL metrics
    all_real_images = []
    all_generated_images = {t: [] for t in timesteps}

    with torch.no_grad():
        # Process ALL validation data - only collect images
        for images, labels in dataloader:
            images = images.to(accelerator.device)
            labels = labels.to(accelerator.device)

            for img_idx in range(images.shape[0]):
                single_image = images[img_idx:img_idx+1]
                single_label = labels[img_idx:img_idx+1]

                # Tokenize to get latent tokens
                cls_token, latent_tokens, _ = model.tokenize(single_image, single_label)

                # Store original image
                all_real_images.append(single_image[0])

                # For each timestep, create reconstructed image
                for timestep in timesteps:
                    # Add noise to image
                    t_tensor = torch.full((1,), timestep, device=accelerator.device)
                    noise = torch.randn_like(single_image)
                    noisy_image = noise_scheduler.add_noise(single_image, noise, t_tensor)

                    # Denoise pass
                    output_image = model.denoise(
                        noisy_image, cls_token, latent_tokens,
                        t_tensor, num_masked_tokens=None
                    )

                    # Store generated image
                    all_generated_images[timestep].append(output_image[0])

    # Compute all metrics from collected data
    # Stack real images once (shared across all timesteps)
    real_batch = torch.stack(all_real_images)

    avg_metrics = {t: {} for t in timesteps}

    for timestep in timesteps:
        gen_batch = torch.stack(all_generated_images[timestep])

        for metric in metrics_list:
            if metric == 'mse':
                avg_metrics[timestep][metric] = F.mse_loss(gen_batch, real_batch).item()
            elif metric == 'psnr':
                mse = F.mse_loss(gen_batch, real_batch).item()
                psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse)))
                avg_metrics[timestep][metric] = psnr
            elif metric == 'fid':
                from tools.metrics import compute_fid
                fid_score = compute_fid(real_batch, gen_batch)
                avg_metrics[timestep][metric] = fid_score

    return {'metrics_per_timestep': avg_metrics}
```

**Step 3: Commit**

```bash
git add src/evaluators/stage1_evaluator.py
git commit -m "feat(evaluator): implement timestep sweep evaluation (separated viz and metrics)"
```

---

## Task 5: Implement Attention Mask Sweep Evaluation

**Files:**
- Modify: `src/evaluators/stage1_evaluator.py`

**Step 1: Implement visualize_attn_mask_sweep function**

Add to `src/evaluators/stage1_evaluator.py`:

```python
def visualize_attn_mask_sweep(
    model, accelerator, dataloader, noise_scheduler,
    config, save_dir, epoch, project_name=None
):
    """
    Generate visualization for reconstruction from pure noise with different masked token counts.

    Process:
    1. Select N images from first batch
    2. For each mask_token_num:
       - Generate pure noise
       - Run denoise pass with num_masked_tokens
       - Append reconstructed image
    3. Build grid with visualizer

    Args:
        model: Poet model
        accelerator: Accelerator instance
        dataloader: Validation data loader
        noise_scheduler: Noise scheduler
        config: Dict with 'mask_token_nums' and 'num_vis_samples'
        save_dir: Directory to save visualizations
        epoch: Current epoch
        project_name: Optional project name

    Returns:
        None (saves visualization to disk)
    """
    from tools.visualizers import create_grid_visualizer
    import os

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    mask_token_nums = config.get('mask_token_nums', [0, 32, 64, 96, 128, 160, 192])
    num_vis_samples = config.get('num_vis_samples', 8)

    images_2d = []

    with torch.no_grad():
        # Only process first batch for visualization
        images, labels = next(iter(dataloader))
        images = images.to(accelerator.device)
        labels = labels.to(accelerator.device)

        for img_idx in range(min(num_vis_samples, images.shape[0])):
            single_image = images[img_idx:img_idx+1]
            single_label = labels[img_idx:img_idx+1]

            # Tokenize to get latent tokens
            cls_token, latent_tokens, _ = model.tokenize(single_image, single_label)

            # Row for this image
            row_images = []

            # Column 0: Original image
            row_images.append(single_image[0])

            # Column 1: Pure noise
            noise_image = torch.randn_like(single_image)
            row_images.append(noise_image[0])

            # Column 2+: Reconstructed images with different num_masked_tokens
            # Use t=1.0 for pure noise
            t_tensor = torch.full((1,), 1.0, device=accelerator.device)
            for mask_num in mask_token_nums:
                # Denoise pass with num_masked_tokens
                # None means no masking, 0 means all tokens masked
                num_masked = None if mask_num == 0 else mask_num
                output_image = model.denoise(
                    noise_image, cls_token, latent_tokens,
                    t_tensor, num_masked_tokens=num_masked
                )

                # Append reconstructed image
                row_images.append(output_image[0])

            images_2d.append(row_images)

    # Create visualization
    xticks = ["original", "noise"] + [str(m) for m in mask_token_nums]
    save_path = os.path.join(save_dir, f'attn_mask_sweep_epoch{epoch:04d}.png')

    create_grid_visualizer(
        image_name="attn_mask_sweep",
        xlabel="",
        ylabel="",
        xticks=xticks,
        yticks=[],
        images_2d=images_2d,
        save_path=save_path,
        project_name=project_name
    )
```

**Step 2: Implement calculate_attn_mask_sweep_metrics function**

Add to `src/evaluators/stage1_evaluator.py`:

```python
def calculate_attn_mask_sweep_metrics(
    model, accelerator, dataloader, noise_scheduler,
    config
):
    """
    Calculate metrics for reconstruction from pure noise with different masked token counts.

    Process:
    1. Iterate over ALL validation data
    2. For each mask_token_num:
       - Generate pure noise
       - Run denoise pass with num_masked_tokens
       - Collect real and generated images
    3. Compute all metrics from collected data

    Args:
        model: Poet model
        accelerator: Accelerator instance
        dataloader: Validation data loader
        noise_scheduler: Noise scheduler
        config: Dict with 'mask_token_nums' and 'metrics' list

    Returns:
        Dict with metrics per mask count
    """
    import torch.nn.functional as F

    model.eval()

    mask_token_nums = config.get('mask_token_nums', [0, 32, 64, 96, 128, 160, 192])
    metrics_list = config.get('metrics', ['mse', 'psnr'])

    # Collect real and generated images for ALL metrics
    all_real_images = []
    all_generated_images = {m: [] for m in mask_token_nums}

    with torch.no_grad():
        # Process ALL validation data - only collect images
        for images, labels in dataloader:
            images = images.to(accelerator.device)
            labels = labels.to(accelerator.device)

            for img_idx in range(images.shape[0]):
                single_image = images[img_idx:img_idx+1]
                single_label = labels[img_idx:img_idx+1]

                # Tokenize to get latent tokens
                cls_token, latent_tokens, _ = model.tokenize(single_image, single_label)

                # Store original image (for comparison with reconstructed)
                all_real_images.append(single_image[0])

                # For each mask count, create reconstructed image from pure noise
                for mask_num in mask_token_nums:
                    # Generate pure noise (like noise recovery)
                    noise_image = torch.randn_like(single_image)

                    # Use t=1.0 for pure noise
                    t_tensor = torch.full((1,), 1.0, device=accelerator.device)

                    # Denoise pass with num_masked_tokens
                    # None means no masking, 0 means all tokens masked
                    num_masked = None if mask_num == 0 else mask_num
                    output_image = model.denoise(
                        noise_image, cls_token, latent_tokens,
                        t_tensor, num_masked_tokens=num_masked
                    )

                    # Store generated image
                    all_generated_images[mask_num].append(output_image[0])

    # Compute all metrics from collected data
    # Stack real images once (shared across all mask counts)
    real_batch = torch.stack(all_real_images)

    avg_metrics = {m: {} for m in mask_token_nums}

    for mask_num in mask_token_nums:
        gen_batch = torch.stack(all_generated_images[mask_num])

        for metric in metrics_list:
            if metric == 'mse':
                avg_metrics[mask_num][metric] = F.mse_loss(gen_batch, real_batch).item()
            elif metric == 'psnr':
                mse = F.mse_loss(gen_batch, real_batch).item()
                psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse)))
                avg_metrics[mask_num][metric] = psnr
            elif metric == 'fid':
                from tools.metrics import compute_fid
                fid_score = compute_fid(real_batch, gen_batch)
                avg_metrics[mask_num][metric] = fid_score

    return {'metrics_per_mask': avg_metrics}
```

**Step 3: Commit**

```bash
git add src/evaluators/stage1_evaluator.py
git commit -m "feat(evaluator): implement attention mask sweep evaluation (separated viz and metrics)"
```

---

## Task 6: Update Main Evaluator Class

**Files:**
- Modify: `src/evaluators/stage1_evaluator.py`

**Step 1: Update Stage1Evaluator.__init__ method**

Find the `__init__` method (around line 64) and update:

```python
def __init__(self, model, accelerator, evaluation_dir, eval_config, project_name=None):
    """
    初始化 Stage1Evaluator

    Args:
        model: Poet 模型
        accelerator: Accelerator 实例
        evaluation_dir: 评估结果保存目录
        eval_config: Evaluation config dict with visualization settings
        project_name: 项目名称（可选，用于可视化）
    """
    self.model = model
    self.accelerator = accelerator
    self.evaluation_dir = evaluation_dir
    self.eval_config = eval_config
    self.project_name = project_name
```

**Step 2: Replace the evaluate method**

Replace the entire `evaluate` method (lines 79-288) with:

```python
def evaluate(self, dataloader, noise_scheduler, epoch, log_writer=None):
    """
    使用 EMA 模型评估 Stage 1 重建质量

    根据配置运行三种评估模式的子集

    Args:
        dataloader: 数据加载器
        noise_scheduler: Noise scheduler
        epoch: 当前 epoch
        log_writer: TensorBoard writer

    Returns:
        metrics: 字典，包含评估指标
    """
    self.model.eval()

    # 1. 保存当前训练参数
    train_state_dict = {k: v.clone().detach() for k, v in self.model.state_dict().items()}

    # 2. 加载 EMA1 参数
    ema_state_dict = self.model.store_ema(which=1)
    if ema_state_dict is None:
        print("⚠️  EMA not initialized, skipping evaluation")
        self.model.train()
        return {}

    self.model.load_ema(ema_state_dict)
    print(f"✅ Switched to EMA1 for evaluation at epoch {epoch}")

    try:
        results = {}
        metrics_config = self.eval_config.get('metrics', {})
        viz_config = self.eval_config.get('visualization', {})

        # Get metrics configuration (global for all evaluations)
        metrics_list = metrics_config.get('types', ['mse', 'psnr'])
        metrics_enabled = metrics_config.get('enabled', True)

        # Visualization pass (only on first batch, num_vis_samples images)
        if viz_config.get('noise_recovery', {}).get('enabled', False):
            print("🎨 Visualizing noise recovery...")
            config = viz_config['noise_recovery']
            visualize_noise_recovery(
                self.model, self.accelerator, dataloader, noise_scheduler,
                config, self.evaluation_dir, epoch, self.project_name
            )

        if viz_config.get('timestep_sweep', {}).get('enabled', False):
            print("🎨 Visualizing timestep sweep...")
            config = viz_config['timestep_sweep']
            visualize_timestep_sweep(
                self.model, self.accelerator, dataloader, noise_scheduler,
                config, self.evaluation_dir, epoch, self.project_name
            )

        if viz_config.get('attn_mask_sweep', {}).get('enabled', False):
            print("🎨 Visualizing attention mask sweep...")
            config = viz_config['attn_mask_sweep']
            visualize_attn_mask_sweep(
                self.model, self.accelerator, dataloader, noise_scheduler,
                config, self.evaluation_dir, epoch, self.project_name
            )

        # Metrics calculation pass (on ALL validation data)
        if metrics_enabled:
            print("📊 Computing metrics on entire validation set...")

            if viz_config.get('noise_recovery', {}).get('enabled', False):
                print("  - noise_recovery metrics")
                config = viz_config['noise_recovery'].copy()
                config['metrics'] = metrics_list
                results['noise_recovery'] = calculate_noise_recovery_metrics(
                    self.model, self.accelerator, dataloader, noise_scheduler,
                    config
                )

            if viz_config.get('timestep_sweep', {}).get('enabled', False):
                print("  - timestep_sweep metrics")
                config = viz_config['timestep_sweep'].copy()
                config['metrics'] = metrics_list
                results['timestep_sweep'] = calculate_timestep_sweep_metrics(
                    self.model, self.accelerator, dataloader, noise_scheduler,
                    config
                )

            if viz_config.get('attn_mask_sweep', {}).get('enabled', False):
                print("  - attn_mask_sweep metrics")
                config = viz_config['attn_mask_sweep'].copy()
                config['metrics'] = metrics_list
                results['attn_mask_sweep'] = calculate_attn_mask_sweep_metrics(
                    self.model, self.accelerator, dataloader, noise_scheduler,
                    config
                )

        # Log results to TensorBoard
        if log_writer is not None and self.accelerator.is_main_process:
            for eval_name, eval_results in results.items():
                for metric_name, metric_value in eval_results.items():
                    if isinstance(metric_value, dict):
                        for key, val in metric_value.items():
                            log_writer.add_scalar(
                                f'eval/{eval_name}/{metric_name}/{key}',
                                val, epoch
                            )
                    else:
                        log_writer.add_scalar(
                            f'eval/{eval_name}/{metric_name}',
                            metric_value, epoch
                        )

    finally:
        # 切换回训练参数
        self.model.load_state_dict(train_state_dict)
        print("✅ Switched back to training parameters")
        self.model.train()

    return results
```

**Step 3: Commit**

```bash
git add src/evaluators/stage1_evaluator.py
git commit -m "refactor(evaluator): update main class to support new evaluation modes"
```

---

## Task 7: Update Trainer to Pass Eval Config

**Files:**
- Modify: `src/trainers/stage1_trainer.py`

**Step 1: Update Stage1Trainer.__init__ to accept eval_config**

Find where `self.evaluator` is initialized (around line 98-110) and update:

```python
# Initialize evaluator
self.evaluator = Stage1Evaluator(
    model, accelerator,
    training_config.checkpoint_dir,
    training_config,  # Pass full training_config which has evaluation section
    project_name
)
```

**Step 2: Verify config is accessible**

The `training_config` object should have the `evaluation` attribute from the YAML config.

**Step 3: Commit**

```bash
git add src/trainers/stage1_trainer.py
git commit -m "fix(trainer): pass evaluation config to evaluator"
```

---

## Task 9: Final Verification

**Step 1: Verify config loads correctly**

```bash
python -c "
from src.utils.config_manager import get_config
config = get_config('configs/stage1/base.yaml')
print('✅ Config loaded successfully')
print('Evaluation config:', config.evaluation)
"
```

**Step 2: Check for syntax errors**

```bash
python -m py_compile src/evaluators/tools/visualizers.py
python -m py_compile src/evaluators/stage1_evaluator.py
python -m py_compile src/trainers/stage1_trainer.py
```

**Step 3: Verify imports work**

```bash
python -c "
from src.evaluators.tools.visualizers import create_grid_visualizer, concatenate_with_arrow
from src.evaluators.stage1_evaluator import evaluate_noise_recovery, evaluate_timestep_sweep, evaluate_attn_mask_sweep
print('✅ All imports successful')
"
```

**Step 4: Final commit**

```bash
git add .
git commit -m "feat: complete Stage 1 evaluator redesign"
```

---

## Summary

This implementation plan:

✅ Separates visualization logic from evaluation logic
✅ Implements three independent evaluation functions
✅ Config-driven enable/disable via YAML
✅ Clean visualizer interface with flexible grids
✅ Arrow concatenation for input→output pairs
✅ Minimal metrics (visualization-focused)
✅ Comprehensive documentation
✅ Proper error handling and logging

**Total estimated implementation time:** 1-2 hours

**Key files modified:**
- `configs/stage1/base.yaml` - Configuration structure
- `src/evaluators/tools/visualizers.py` - Visualization functions
- `src/evaluators/stage1_evaluator.py` - Evaluation functions
- `src/trainers/stage1_trainer.py` - Trainer integration

**Next steps after implementation:**
- Test with real model
- Adjust visual spacing/arrow styling if needed
- Add additional metrics if required
- Performance optimization if needed
