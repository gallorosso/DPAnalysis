import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

class BasePlotter:
    """
    Base class for all plotters with common utilities and styling.
    """
    
    def __init__(self, style_config=None):
        self.set_plot_style(style_config)
        
    def set_plot_style(self, config=None):
        """Set consistent matplotlib style across all plots."""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Default style parameters
        self.style_config = {
            'figure.figsize': (12, 8),
            'figure.dpi': 150,
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'lines.linewidth': 1.5,
            'lines.markersize': 4,
            'grid.alpha': 0.3
        }
        
        if config:
            self.style_config.update(config)
            
        plt.rcParams.update(self.style_config)
        
    def create_figure(self, subplots=(1, 1), figsize=None, **kwargs):
        """Create a figure with consistent styling."""
        if figsize is None:
            base_figsize = self.style_config['figure.figsize']
            # Adjust size for multi-panel plots
            figsize = (base_figsize[0] * subplots[1], base_figsize[1] * subplots[0])
            
        return plt.subplots(subplots[0], subplots[1], 
                          figsize=figsize, 
                          dpi=self.style_config['figure.dpi'],
                          **kwargs)
        
    def save_figure(self, fig, filepath, formats=['png'], **kwargs):
        """Save figure in multiple formats."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        for fmt in formats:
            save_path = f"{filepath}.{fmt}"
            fig.savefig(save_path, bbox_inches='tight', 
                       dpi=self.style_config['figure.dpi'], **kwargs)
            print(f"Saved: {save_path}")
            
    def close_figure(self, fig):
        """Close figure to free memory."""
        plt.close(fig)
        
    def set_axis_limits(self, ax, xlim=None, ylim=None):
        """Set axis limits consistently."""
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
            
    def add_grid(self, ax):
        """Add grid to axis."""
        ax.grid(True, alpha=self.style_config['grid.alpha'])