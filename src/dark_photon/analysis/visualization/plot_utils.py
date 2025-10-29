import numpy as np
import matplotlib.pyplot as plt

def set_plot_style(style_dict=None):
    """Set matplotlib plotting style."""
    default_style = {
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'legend.frameon': True,
        'legend.framealpha': 0.9
    }
    
    if style_dict:
        default_style.update(style_dict)
        
    plt.rcParams.update(default_style)

def dB_to_linear(dB_value):
    """Convert dB value to linear scale."""
    return 10**(dB_value / 10)

def linear_to_dB(linear_value):
    """Convert linear value to dB scale."""
    return 10 * np.log10(linear_value)

def create_color_cycle():
    """Create consistent color cycle for plots."""
    return plt.rcParams['axes.prop_cycle'].by_key()['color']