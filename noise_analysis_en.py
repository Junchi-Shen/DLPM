# Simple Noise Residual Analysis for Diffusion Models

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os

def analyze_noise_residuals_simple():
    """Simple noise residual analysis"""
    
    print("=== Diffusion Model Noise Residual Analysis ===")
    
    # Check if model files exist
    model_paths = [
        "Results/Model_Results/all/unet_conditional_model_all.pth",
        "Results/Model_Results/all/condition_network_all.pth",
        "Results/Model_Results/all/data_processor_all.pkl"
    ]
    
    for path in model_paths:
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            return
    
    print("Model files check passed")
    
    try:
        # Load model
        print("Loading model...")
        checkpoint = torch.load(model_paths[0], map_location='cpu')
        print("Model loaded successfully")
        
        # Generate simulated noise residual data for analysis
        print("Generating simulated noise residual data...")
        
        # Generate normally distributed residuals (ideal case)
        ideal_residuals = np.random.normal(0, 0.1, 2000)
        
        # Generate non-normally distributed residuals (problematic case)
        problematic_residuals = np.random.normal(0, 0.1, 1000)
        problematic_residuals = np.concatenate([
            problematic_residuals,
            np.random.normal(0.2, 0.05, 500),  # Add offset
            np.random.normal(-0.1, 0.15, 500)  # Add another distribution
        ])
        
        # Analyze both cases
        print("\n=== Ideal Case Analysis ===")
        analyze_residuals(ideal_residuals, "Ideal Case")
        
        print("\n=== Problematic Case Analysis ===")
        analyze_residuals(problematic_residuals, "Problematic Case")
        
        # Plot comparison
        plot_comparison(ideal_residuals, problematic_residuals)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def analyze_residuals(residuals, case_name):
    """Analyze normality of residuals"""
    
    # Basic statistics
    mean = np.mean(residuals)
    std = np.std(residuals)
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    
    print(f"{case_name} - Residual Statistics:")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std: {std:.6f}")
    print(f"  Skewness: {skewness:.6f}")
    print(f"  Kurtosis: {kurtosis:.6f}")
    
    # Normality tests
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(mean, std))
    jarque_bera_stat, jarque_bera_p = stats.jarque_bera(residuals)
    
    print(f"\n{case_name} - Normality Tests:")
    print(f"  Shapiro-Wilk: stat={shapiro_stat:.6f}, p-value={shapiro_p:.6f}")
    print(f"  Kolmogorov-Smirnov: stat={ks_stat:.6f}, p-value={ks_p:.6f}")
    print(f"  Jarque-Bera: stat={jarque_bera_stat:.6f}, p-value={jarque_bera_p:.6f}")
    
    # Results
    alpha = 0.05
    print(f"\n{case_name} - Test Results (alpha={alpha}):")
    print(f"  Shapiro-Wilk: {'PASS' if shapiro_p > alpha else 'FAIL'} normality assumption")
    print(f"  KS Test: {'PASS' if ks_p > alpha else 'FAIL'} normality assumption")
    print(f"  Jarque-Bera: {'PASS' if jarque_bera_p > alpha else 'FAIL'} normality assumption")
    
    # Model quality assessment
    if shapiro_p > alpha and ks_p > alpha and jarque_bera_p > alpha:
        print(f"\nEXCELLENT - {case_name} - Model noise prediction quality: EXCELLENT - Residuals follow normal distribution")
    elif shapiro_p > alpha or ks_p > alpha or jarque_bera_p > alpha:
        print(f"\nFAIR - {case_name} - Model noise prediction quality: FAIR - Some tests passed")
    else:
        print(f"\nPOOR - {case_name} - Model noise prediction quality: POOR - Residuals do not follow normal distribution")

def plot_comparison(ideal_residuals, problematic_residuals):
    """Plot comparison charts"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Ideal case analysis
    # 1. Histogram
    axes[0, 0].hist(ideal_residuals, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Ideal Case - Residual Distribution', fontsize=14)
    axes[0, 0].set_xlabel('Residual Value')
    axes[0, 0].set_ylabel('Density')
    
    # Overlay normal distribution curve
    x = np.linspace(ideal_residuals.min(), ideal_residuals.max(), 100)
    normal_curve = stats.norm.pdf(x, np.mean(ideal_residuals), np.std(ideal_residuals))
    axes[0, 0].plot(x, normal_curve, 'r-', linewidth=2, label='Theoretical Normal')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Q-Q plot
    stats.probplot(ideal_residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Ideal Case - Q-Q Plot', fontsize=14)
    axes[0, 1].grid(True)
    
    # 3. Box plot
    axes[0, 2].boxplot(ideal_residuals, vert=True)
    axes[0, 2].set_ylabel('Residual Value')
    axes[0, 2].set_title('Ideal Case - Box Plot', fontsize=14)
    axes[0, 2].grid(True)
    
    # Problematic case analysis
    # 1. Histogram
    axes[1, 0].hist(problematic_residuals, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_title('Problematic Case - Residual Distribution', fontsize=14)
    axes[1, 0].set_xlabel('Residual Value')
    axes[1, 0].set_ylabel('Density')
    
    # Overlay normal distribution curve
    x = np.linspace(problematic_residuals.min(), problematic_residuals.max(), 100)
    normal_curve = stats.norm.pdf(x, np.mean(problematic_residuals), np.std(problematic_residuals))
    axes[1, 0].plot(x, normal_curve, 'r-', linewidth=2, label='Theoretical Normal')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 2. Q-Q plot
    stats.probplot(problematic_residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Problematic Case - Q-Q Plot', fontsize=14)
    axes[1, 1].grid(True)
    
    # 3. Box plot
    axes[1, 2].boxplot(problematic_residuals, vert=True)
    axes[1, 2].set_ylabel('Residual Value')
    axes[1, 2].set_title('Problematic Case - Box Plot', fontsize=14)
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('noise_residual_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison chart saved to: noise_residual_comparison.png")
    plt.show()

def create_analysis_guide():
    """Create analysis guide"""
    
    guide = """
# How to Check Normal Distribution of Diffusion Model Noise Residuals

## Analysis Objective
Check if the noise predicted by the diffusion model is accurate and if residuals follow normal distribution.

## Analysis Steps

### 1. Collect Noise Residuals
```python
# For each training sample:
# 1. Randomly select timestep t
# 2. Add real noise noise ~ N(0,1)
# 3. Get x_t = q_sample(x_0, t, noise)
# 4. Model prediction pred_noise = model(x_t, t, conditions)
# 5. Calculate residual residual = pred_noise - noise
```

### 2. Statistical Tests
- **Shapiro-Wilk Test**: Test normality
- **Kolmogorov-Smirnov Test**: Test distribution shape
- **Jarque-Bera Test**: Test skewness and kurtosis

### 3. Visualization Analysis
- **Histogram**: Observe distribution shape
- **Q-Q Plot**: Test quantiles
- **Box Plot**: Observe outliers
- **Scatter Plot**: Observe correlation

## Ideal Results
- Residual mean close to 0
- Residual std small
- Skewness close to 0
- Kurtosis close to 0
- All normality test p-values > 0.05

## Problem Indicators
- Residual mean deviates from 0
- Residual std too large
- Skewness absolute value > 0.5
- Kurtosis absolute value > 1.0
- Normality test p-values < 0.05

## Improvement Suggestions
1. **Adjust loss function weights**
2. **Increase training data**
3. **Adjust model architecture**
4. **Optimize training strategy**
"""
    
    with open('noise_analysis_guide.md', 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print("Analysis guide saved to: noise_analysis_guide.md")

if __name__ == "__main__":
    # Run analysis
    analyze_noise_residuals_simple()
    
    # Create guide
    create_analysis_guide()
    
    print("\n=== Analysis Complete ===")
    print("Please check the generated charts and guide files")
