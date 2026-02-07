# Project: ShapeNN
Context & Project Goal: I am building an Industrial-Grade 2D Shape Matching Network (Translation x, y + Rotation theta).The goal is to achieve sub-pixel accuracy (< 0.5 px) and robust rotation estimation for complex, random 2D polygons.

Constraints & Requirements:
No Dataset Simplification: The model must handle random 3-10 sided polygons. Do not simplify this to triangles or fixed shapes. The goal is robustness to arbitrary industrial parts.

No Regression Heads: We previously tried using a dense layer to regress sin, cos, for rotation but it failed (converged to 0,0). We have abandoned direct regression. Only try this if current approach fails.

Approach: We are using a Discriminative Siamese UNet with a Dense Search Inference strategy.
Training: The network is trained as a binary discriminator.
Input: Template (Distance Field) + Query (Image).
Positive Pair: If Template and Query are aligned (within $\pm 5^\circ$), output a Gaussian Heatmap at the target location.
Negative Pair: If Template and Query are rotated differently ($> 20^\circ$), output Zero (Black Image).

Inference: We rotate the template 36 times (batch of 36), run them all through the UNet, and pick the one with the highest heatmap activation.

Sub-Pixel Strategy: We use cv2.LINE_AA and GaussianBlur in data generation to create smooth gradients, and we use a "Center of Mass" calculation on the predicted heatmap peak to recover sub-pixel coordinates.

You are free to change the architecture of the network as long as it follows the requirements and achieves the goal.

### Environment Rules
- **Python Environment:** Always use the active `nn_env` virtual environment.
- **Hardware:** Use CUDA if available, the laptop has an RTX 2060 6GB GPU.
- **Style:** Use PyTorch for the model definition.
- **Data:** Training data can be generated using the `generate_polygon_dataset.py` script.

### Success Criteria
- The model is considered "good" when validation accuracy exceeds 95%.
- If a training run fails, analyze the logs and propose a fix before re-running.
- If the model struggles, tune the Weighted Loss (currently 50x) or the Discriminative Mix (currently 50/50) or change the loss as you see fit.
- Do NOT suggest simplifying the dataset to triangles. Robustness to complex shapes is non-negotiable.


### Code Backup
Make sure to regularly commit and push the code to the git repository - https://github.com/MythraV/shape_match_NN.git