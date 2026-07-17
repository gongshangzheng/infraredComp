# Stage 1 Evaluator Redesign

**Date:** 2026-03-07
**Status:** Approved Design

## Overview

Redesign the Stage 1 evaluator to support three distinct visualization modes:
1. **Noise Recovery**: Progressive n-step reconstruction from pure noise
2. **Timestep Sweep**: Single reconstruction at different initial timesteps
3. **Attention Mask Sweep**: Reconstruction with varying masked token counts

## Architecture

### High-Level Structure

The evaluator is restructured into three main components:

1. **Visualizer Module** (`tools/visualizers.py` - new functions)
   - `create_grid_visualizer()`: Clean interface for flexible grid layouts
   - `concatenate_with_arrow()`: Horizontal concatenation with directional arrow
   - Pure visualization logic, no business logic

2. **Evaluation Functions** (in `stage1_evaluator.py`)
   - `evaluate_noise_recovery()`: Handles evaluation type 1
   - `evaluate_timestep_sweep()`: Handles evaluation type 2
   - `evaluate_attn_mask_sweep()`: Handles evaluation type 3
   - Each function is independent and can be called standalone

3. **Configuration** (in `base.yaml`)
   - Config-driven enable/disable for each evaluation type
   - Easy toggling without code changes

### Configuration Structure

```yaml
evaluation:
  eval_every_steps: 10000
  visualization:
    num_vis_samples: 8  # Number of images to show vertically

    # Evaluation type 1: Noise recovery
    noise_recovery:
      enabled: true
      num_steps: 4  # n steps for reconstruction

    # Evaluation type 2: Timestep sweep
    timestep_sweep:
      enabled: true
      initial_timesteps: [0.25, 0.5, 0.75, 1.0]

    # Evaluation type 3: Attention mask sweep
    attn_mask_sweep:
      enabled: true
      mask_token_nums: [0, 32, 64, 96, 128, 160, 192]
```

## Visualizer Implementation

### Main Interface

```python
def create_grid_visualizer(
    image_name: str,
    xlabel: str,  # Can be empty string ""
    ylabel: str,  # Can be empty string ""
    xticks: List[str],  # ["original", "noise", "step1", "step2", ...]
    yticks: List[str],  # Can be empty []
    images_2d: List[List[torch.Tensor]],  # 2D list [row][col]
    save_path: str,
    project_name: Optional[str] = None
) -> None
```

**Features:**
- **Axis flexibility**: Empty strings for xlabel/ylabel hide axis labels
- **Automatic sizing**: Grid dimensions inferred from `images_2d` shape
- **No business logic**: Just rendering what it receives
- **Tensor handling**: Accepts PyTorch tensors, handles CPU/clipping internally

### Helper Functions

1. **`concatenate_with_arrow(input_img, output_img, gap_size=8)`**
   - Concatenates two images horizontally with a gap and arrow
   - Arrow points left → right (input → reconstruction)
   - Returns single concatenated image tensor
   - Used by eval2 and eval3 for "input → output" pairs

2. **`images_to_grid(images_2d, xticks, yticks, xlabel, ylabel)`**
   - Internal helper for matplotlib grid layout
   - Handles automatic spacing and alignment

## Evaluation Functions

### Evaluation 1: Noise Recovery

**Purpose**: Demonstrate progressive reconstruction from pure noise

**Process:**
1. Select N images from validation set
2. For each image:
   - Store original
   - Generate pure noise (`torch.randn_like`)
   - Run n forward passes (configurable, default 4)
   - Save each intermediate output
3. Build grid:
   - **Horizontal**: [original, noise, step1, step2, ..., stepN]
   - **Vertical**: N different images
4. Call visualizer

**Horizontal labels**: `["original", "noise", "step1", "step2", ..., "stepN"]`
**Vertical labels**: None (empty)
**Axis names**: None (empty strings)

**Key Details:**
- Initial state: Pure noise
- Progressive refinement: Step 1 output → step 2 input
- Metrics: Average MSE at each step

### Evaluation 2: Timestep Sweep

**Purpose**: Show reconstruction quality at different noise levels

**Process:**
1. Select N images from validation set
2. For each timestep t in config:
   - Add noise at level t: `noisy = scheduler.add_noise(img, noise, t)`
   - Run single forward pass
   - Concatenate: [noisy → arrow → reconstructed]
3. Build grid:
   - **Column 0**: original image (no arrow)
   - **Columns 1-N**: [noisy@t → reconstructed] pairs
4. Call visualizer

**Horizontal labels**: `["original", "0.25", "0.5", "0.75", "1.0"]`
**Vertical labels**: None
**Axis names**: None

**Key Details:**
- Uses `concatenate_with_arrow()` for each timestep pair
- First column is special: just original, no arrow
- Metrics: MSE, PSNR per timestep

### Evaluation 3: Attention Mask Sweep

**Purpose**: Evaluate reconstruction with varying context (masked tokens)

**Process:**
1. Select N images from validation set
2. For each mask_token_num in config:
   - Create attention mask: binary mask of shape `[num_latent_tokens]`
     - First `mask_token_num` entries = 0 (masked)
     - Remaining entries = 1 (visible)
   - Run single forward pass with attn_mask
   - Concatenate: [masked_input → arrow → reconstructed]
3. Build grid:
   - **Column 0**: original image (no arrow, no mask)
   - **Columns 1-N**: [masked → reconstructed] pairs
4. Call visualizer

**Horizontal labels**: `["original", "0", "32", "64", "96", "128", "160", "192"]`
**Vertical labels**: None
**Axis names**: None

**Key Details:**
- `mask_token_num=0`: No masking (full attention)
- Pass `attn_mask` parameter to model forward pass
- Metrics: MSE, PSNR per mask count

## Main Evaluator Coordination

```python
class Stage1Evaluator:
    def __init__(self, model, accelerator, evaluation_dir, eval_config, project_name=None):
        self.eval_config = eval_config

    def evaluate(self, dataloader, noise_scheduler, epoch, log_writer=None):
        # Switch to EMA parameters
        results = {}

        if self.eval_config.noise_recovery.enabled:
            results['noise_recovery'] = self.evaluate_noise_recovery(...)

        if self.eval_config.timestep_sweep.enabled:
            results['timestep_sweep'] = self.evaluate_timestep_sweep(...)

        if self.eval_config.attn_mask_sweep.enabled:
            results['attn_mask_sweep'] = self.evaluate_attn_mask_sweep(...)

        # Switch back to training params
        return results
```

**Integration Points:**
- Each evaluation function is independent
- Results dictionary with keys matching config names
- Error handling: One eval failure doesn't stop others
- Metrics: Minimal (visualization-focused)

## Implementation Notes

### Backward Compatibility
- Old `save_reconstruction_comparison()` preserved for existing code
- New visualizer functions added, not replacing

### Model Forward Pass Requirements
- Model must accept `attn_mask` parameter for eval3
- Current model interface may need extension

### Memory Management
- Process images in batches to avoid OOM
- Use `torch.no_grad()` for all evaluations
- Clear intermediate tensors after each eval

### File Organization
```
src/evaluators/
├── stage1_evaluator.py       # Main evaluator with 3 eval functions
└── tools/
    └── visualizers.py        # New grid visualizer + helpers
```

## Success Criteria

1. ✅ Config-driven enable/disable of each evaluation type
2. ✅ Clean visualizer interface with flexible grid layouts
3. ✅ Three distinct visualization modes working independently
4. ✅ Minimal metrics (visualization-focused)
5. ✅ Arrow concatenation for input→output pairs
6. ✅ Proper axis labels and grid formatting

## Next Steps

1. Implement `create_grid_visualizer()` and helper functions
2. Implement `evaluate_noise_recovery()`
3. Implement `evaluate_timestep_sweep()`
4. Implement `evaluate_attn_mask_sweep()`
5. Update configuration structure in base.yaml
6. Integrate with main evaluator class
7. Test with sample data
