# Unified AI Lab & Analysis Platform

**Academic Defense Report & Setup Guide**

## 1. Architecture Justification

### 1.1 Module A: 2D Vision (ResNet50 / EfficientNet)
For Document Forgery and visual texture tasks, capturing fine-grained structural differences and spatial locality is critical. We elected to use **ResNet50** and **EfficientNet** due to their proven feature extraction capabilities in capturing varying scales of image details. 
- **ResNet50:** The residual connections prevent vanishing gradients, allowing us to capture deep hierarchical features essential for identifying subtle anomalies in high-quality document forgeries.
- **EfficientNet:** Utilized for a computationally efficient alternative when deployment scale requires faster inference without compromising spatial detail extraction.

### 1.2 Module B: 1D Spectroscopy (1D-CNN)
In chemistry domains, such as FT-IR data, the spectral curves exhibit strong "local" correlations—peaks and valleys occur in contiguous spectral bands.
- **1D-CNN:** Chosen over dense networks because it inherently exploits this locality through 1-dimensional convolutions.
- **Adjustable Parameters:** Providing dynamic API adjustments for `kernel_size` and `dropout` enables users to test varying degrees of feature locality (e.g., broad vs. narrow spectral peaks) and regularization empirically, preventing overfitting on noisy chemical datasets.

### 1.3 Module C: Math & Sequences (MLP/RNN)
For economic and mathematical modeling, sequential dependencies and function approximations dictate the architecture.
- **MLP/RNN Sandbox:** Provides a flexible environment where the user can dynamically swap activations (ReLU, Tanh, Sigmoid) and define hidden layer topologies. This adaptability ensures the system can handle both static functional mappings and temporal sequence tasks (e.g., economic time-series forecasting).

---

## 2. Methodology: Transfer Learning & 2-Stage Training

Our Vision module implements a formal **2-stage Transfer Learning strategy**:
1. **Stage 1 (Feature Extraction):** The base layers (trained on ImageNet) are frozen. Only the newly initialized final fully-connected (FC) layer is trained. This forces the model to linearly separate the target classes based on generic features, minimizing the risk of catastrophic forgetting.
2. **Stage 2 (Fine-tuning):** The base layers are unfrozen, and a drastically reduced learning rate is applied. This allows the network to gently adjust its deep, generic feature maps to be more domain-specific (e.g., focusing on specific ink textures rather than generic natural image edges).

---

## 3. Results Interpretation

The platform is equipped to present Human-in-the-loop insights dynamically. When reviewing failure cases, consider the following templates based on the domain:

### 3.1 Document Forgery (Vision)
- **High-Quality Forgery Failures:** If the model misclassifies a high-quality forgery, consult the **Grad-CAM** output. If the heatmap highlights generic regions rather than specific signatures or stamps, the model may be overfitting to paper texture rather than structural ink anomalies. Consider increasing data augmentation (noise/rotations).

### 3.2 FT-IR Spectroscopy (1D-CNN)
- **Spectral Band Misinterpretation:** If the model struggles to differentiate similar chemical groups, it might be due to an overly large `kernel_size` smoothing out narrow peaks. Adjusting the API to use a smaller `kernel_size` ensures sharp, distinct peaks (e.g., C=O stretches) are retained in the feature maps.

### 3.3 Economic Time-Series (Math/RNN)
- **Vanishing Gradients & ROI:** If long-term dependencies in the sequence are failing to inform ROI predictions, inspect the selected activation function. Swapping `sigmoid` for `relu` or utilizing an LSTM/GRU variant in the sandbox can alleviate gradient decay.

---

## 4. Setup Guide

### 4.1 Prerequisites
- Python 3.9+
- A compatible GPU (NVIDIA) with CUDA installed (highly recommended for Vision tasks).

### 4.2 Environment Setup
1. Clone this repository and navigate to the root directory.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 4.3 Running the Application
Start the FastAPI backend server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The unified API documentation (Swagger UI) will be available at: `http://localhost:8000/docs`.

### 4.4 Data Augmentation Pipeline
The system integrates natively with PyTorch's `torchvision.transforms` for robust data augmentation. When feeding data into the `/vision` endpoints in production, ensure the preprocessing pipeline includes rotations, random noise injections, and normalization to match the ImageNet statistics expected by the base models.
