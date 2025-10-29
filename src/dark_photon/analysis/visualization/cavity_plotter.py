import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .base_plotter import BasePlotter
from .plot_utils import dB_to_linear, linear_to_dB

class CavityPlotter(BasePlotter):
    """
    Plotter for cavity transmission and reflection analysis.
    """
    
    def __init__(self, style_config=None):
        super().__init__(style_config)
        
    def plot_individual_transmission_fit(self, sweep_data, fit_params, fit_range, 
                                       filename, plot_dir, scaleinfo=None):
        """
        Plot individual transmission fit (like in ReadOutCavityTran).
        
        Parameters:
        -----------
        sweep_data : dict
            Contains 'f_GHz_tx', 'I_tx', 'Q_tx', 'f_GHz_tx_big', 'I_tx_big', 'Q_tx_big'
            for both first and second sweeps
        fit_params : tuple
            (bestfitparams1, bestfitparams2) for both sweeps
        fit_range : tuple  
            (rg1, rg2) fit ranges for both sweeps
        filename : str
            Original filename for title
        scaleinfo : object
            Scale info containing xlimit for plotting
        """
        fig, (ax1, ax2) = self.create_figure(subplots=(2, 1))
        
        # Unpack data
        sweep1, sweep2 = sweep_data
        params1, params2 = fit_params
        rg1, rg2 = fit_range
        
        # Lorentzian + linear baseline function
        def lorentzian_linear(a, x):
            return a[0] / (4 * (a[2]**2) * ((x / a[1]) - 1)**2 + 1) + a[3] * x + a[4]
        
        # Plot 1: Full view in dB
        # Big sweep data
        power_big_db = linear_to_dB(sweep1['I_tx_big']**2 + sweep1['Q_tx_big']**2)
        ax1.plot(sweep1['f_GHz_tx_big'], power_big_db, '--k', alpha=0.7, label='full sweep')
        
        # First sweep data and fit
        power1_db = linear_to_dB(sweep1['I_tx']**2 + sweep1['Q_tx']**2)
        ax1.plot(sweep1['f_GHz_tx'], power1_db, 'or', markersize=2, label='before, data')
        
        fit_freq1 = sweep1['f_GHz_tx'][rg1[0]:rg1[1]+1]
        fit_power1_db = linear_to_dB(lorentzian_linear(params1, fit_freq1))
        ax1.plot(fit_freq1, fit_power1_db, '-r', linewidth=2, label='before, fit')
        
        # Second sweep data and fit  
        power2_db = linear_to_dB(sweep2['I_tx2']**2 + sweep2['Q_tx2']**2)
        ax1.plot(sweep2['f_GHz_tx2'], power2_db, 'og', markersize=2, label='after, data')
        
        fit_freq2 = sweep2['f_GHz_tx2'][rg2[0]:rg2[1]+1]
        fit_power2_db = linear_to_dB(lorentzian_linear(params2, fit_freq2))
        ax1.plot(fit_freq2, fit_power2_db, '-g', linewidth=2, label='after, fit')
        
        ax1.set_xlabel('Frequency [GHz]')
        ax1.set_ylabel('Power [dB]')
        ax1.set_title(f'Cavity Transmission: {Path(filename).name}', fontsize=11)
        ax1.legend()
        self.add_grid(ax1)
        
        if scaleinfo and hasattr(scaleinfo, 'xlimit'):
            ax1.set_xlim(scaleinfo.xlimit)
        ax1.set_ylim([-85, -40])
        
        # Plot 2: Normalized detuning view
        tx_peak_norm = np.max(lorentzian_linear(params1, fit_freq1))
        
        # First sweep normalized
        detuning1 = sweep1['f_GHz_tx'] - params1[1]
        power1_norm = (sweep1['I_tx']**2 + sweep1['Q_tx']**2) / tx_peak_norm
        ax2.plot(detuning1, power1_norm, 'or', markersize=2, label='before, data')
        
        fit_detuning1 = fit_freq1 - params1[1]
        fit_power1_norm = lorentzian_linear(params1, fit_freq1) / tx_peak_norm
        ax2.plot(fit_detuning1, fit_power1_norm, '-r', linewidth=2, label='before, fit')
        
        # Second sweep normalized
        detuning2 = sweep2['f_GHz_tx2'] - params1[1]  # Use first fit for reference
        power2_norm = (sweep2['I_tx2']**2 + sweep2['Q_tx2']**2) / tx_peak_norm
        ax2.plot(detuning2, power2_norm, 'og', markersize=2, label='after, data')
        
        fit_detuning2 = fit_freq2 - params1[1]  # Use first fit for reference
        fit_power2_norm = lorentzian_linear(params2, fit_freq2) / tx_peak_norm
        ax2.plot(fit_detuning2, fit_power2_norm, '-g', linewidth=2, label='after, fit')
        
        ax2.set_xlabel('Detuning (f - fit1) [GHz]')
        ax2.set_ylabel('Power [arb]')
        ax2.legend()
        self.add_grid(ax2)
        
        plt.tight_layout()
        
        # Save figure
        save_path = Path(plot_dir) / f"tx_fit_{Path(filename).stem}"
        self.save_figure(fig, save_path)
        
        return fig
    
    def plot_transmission_summary(self, params_avg, params1, params2, scaleinfo, plot_dir):
        """
        Create 6-panel transmission summary plot from ReadOutCavityTran.
        """
        fig, axes = self.create_figure(subplots=(3, 2))
        axes = axes.flatten()
        
        num_iterations = len(params_avg)
        iterations = np.arange(num_iterations)
        
        # Panel 1: Cavity center frequency
        axes[0].plot(iterations, params_avg[:, 1], '.', markersize=12)
        axes[0].set_ylabel('TM$_{010}$ Frequency (GHz)')
        axes[0].set_xlabel('Iteration')
        axes[0].set_title('Cavity Center Frequency')
        self.add_grid(axes[0])
        
        # Panel 2: Tuning step size
        freq_diff = np.diff(params_avg[:, 1]) * 1e6  # kHz
        axes[1].plot(iterations[1:], freq_diff, '.', markersize=12)
        axes[1].set_ylabel('ΔTM$_{010}$ Frequency [kHz]')
        axes[1].set_xlabel('Iteration')
        axes[1].set_title('Tuning Step Size')
        self.add_grid(axes[1])
        
        # Panel 3: Q factor vs frequency
        Q1 = params1[:, 2]
        Q2 = params2[:, 2] 
        Qavg = params_avg[:, 2]
        
        axes[2].plot(params1[:, 1], Q1, '.', markersize=12, label='1st')
        axes[2].plot(params2[:, 1], Q2, '.', markersize=12, label='2nd')
        axes[2].plot(params_avg[:, 1], Qavg, '.', markersize=12, label='Avg.')
        axes[2].set_ylabel('Q Factor')
        axes[2].set_xlabel('TM$_{010}$ Frequency [GHz]')
        axes[2].set_title('Cavity Unloaded Quality Factor')
        axes[2].legend()
        self.add_grid(axes[2])
        
        # Panel 4: Frequency drift
        freq_drift_khz = (params1[:, 1] - params2[:, 1]) * 1e6
        axes[3].plot(params1[:, 1], freq_drift_khz, '.', markersize=12)
        axes[3].set_ylabel('TM$_{010}$ Drift (kHz)')
        axes[3].set_xlabel('TM$_{010}$ Frequency [GHz]')
        axes[3].set_title('Drift in Cavity Frequency')
        self.add_grid(axes[3])
        
        # Panel 5: Transmission peak
        axes[4].plot(params1[:, 1], params1[:, 0], '.', markersize=12, label='tx1')
        axes[4].plot(params2[:, 1], params2[:, 0], '.', markersize=12, label='tx2')
        axes[4].set_ylabel('TX Peak')
        axes[4].set_xlabel('TM$_{010}$ Frequency [GHz]')
        axes[4].set_title('TX Peak')
        axes[4].legend()
        self.add_grid(axes[4])
        
        # Panel 6: Transmission baseline
        baseline1 = params1[:, 1] * params1[:, 3] + params1[:, 4]
        baseline2 = params2[:, 1] * params2[:, 3] + params2[:, 4]
        
        axes[5].plot(params1[:, 1], baseline1, '.', markersize=12, label='tx1')
        axes[5].plot(params2[:, 1], baseline2, '.', markersize=12, label='tx2')
        axes[5].set_ylabel('TX Baseline')
        axes[5].set_xlabel('TM$_{010}$ Frequency [GHz]')
        axes[5].set_title('TX Baseline')
        axes[5].legend()
        self.add_grid(axes[5])
        
        plt.tight_layout()
        
        # Save figure
        save_path = Path(plot_dir) / "tx_fitting_results"
        self.save_figure(fig, save_path)
        
        return fig
    
    def plot_individual_reflection_fit(self, data1, data2, fit_params1, fit_params2, 
                                     fit_ranges, beta_vals, filename, plot_dir):
        """
        Plot individual reflection fit with data, fit, and baseline.
        """
        fig, ax = self.create_figure()
        
        # Lorentzian + linear baseline function
        def lorentzian_linear(a, x):
            return a[0] / (4 * (a[2]**2) * ((x / a[1]) - 1)**2 + 1) + a[3] * x + a[4]
        
        def linear_baseline(a, x):
            return a[3] * x + a[4]
        
        # Unpack fit ranges
        rg1, rg2 = fit_ranges
        
        # Calculate baseline values at ±0.2 GHz from resonance
        bl_val1 = (lorentzian_linear(fit_params1, fit_params1[1] - 0.2) + 
                  lorentzian_linear(fit_params1, fit_params1[1] + 0.2)) / 2
        bl_val2 = (lorentzian_linear(fit_params2, fit_params2[1] - 0.2) + 
                  lorentzian_linear(fit_params2, fit_params2[1] + 0.2)) / 2
        
        # Plot big sweep data if available
        if 'I_rfl_big' in data1 and 'f_GHz_rfl_big' in data1:
            power_big = data1['I_rfl_big']**2 + data1['Q_rfl_big']**2
            ax.plot(data1['f_GHz_rfl_big'], power_big, '--k', alpha=0.7, label='full sweep')
        
        # First reflection data and fits
        power1 = data1['I_rfl']**2 + data1['Q_rfl']**2
        ax.plot(data1['f_GHz_rfl'], power1, '-r', linewidth=1, label='data')
        
        # Fit curve
        fit_freq1 = data1['f_GHz_rfl'][rg1[0]:rg1[1]+1]
        fit_power1 = lorentzian_linear(fit_params1, fit_freq1)
        ax.plot(fit_freq1, fit_power1, '-b', linewidth=2, label='best fit')
        
        # Baseline
        baseline1 = linear_baseline(fit_params1, data1['f_GHz_rfl'])
        ax.plot(data1['f_GHz_rfl'], baseline1, '-g', linewidth=1, label='background')
        
        # Baseline point
        ax.plot(fit_params1[1], bl_val1, 'ro', markersize=6, label='baseline point')
        
        # Second reflection data and fits
        power2 = data2['I_rfl2']**2 + data2['Q_rfl2']**2
        ax.plot(data2['f_GHz_rfl2'], power2, '-r', linewidth=1)
        
        fit_freq2 = data2['f_GHz_rfl2'][rg2[0]:rg2[1]+1]
        fit_power2 = lorentzian_linear(fit_params2, fit_freq2)
        ax.plot(fit_freq2, fit_power2, '-b', linewidth=2)
        
        baseline2 = linear_baseline(fit_params2, data2['f_GHz_rfl2'])
        ax.plot(data2['f_GHz_rfl2'], baseline2, '-g', linewidth=1)
        
        ax.plot(fit_params2[1], bl_val2, 'ro', markersize=6)
        
        ax.set_xlabel('Frequency [GHz]')
        ax.set_ylabel('Power [V²]')
        ax.set_title(f'Reflection: {Path(filename).name}\n' +
                    f'model: Lorentzian + linear baseline, β = {beta_vals[2]:.3f}')
        ax.legend()
        self.add_grid(ax)
        ax.set_ylim([1e-4, 4e-3])
        
        plt.tight_layout()
        
        # Save figure
        save_path = Path(plot_dir) / f"rfl_fit_{Path(filename).stem}"
        self.save_figure(fig, save_path)
        
        return fig
    
    def plot_reflection_summary(self, freq_beta, beta_vals, rfl_fit_params, 
                              scaleinfo, rfl_baseline_db, plot_dir):
        """
        Create 5-panel reflection summary plot from readoutbeta.
        """
        fig, axes = self.create_figure(subplots=(3, 2))
        axes = axes.flatten()
        
        freq_avg = freq_beta[:, 0]
        beta_avg = freq_beta[:, 1]
        
        # Panel 1: Coupling factor (beta)
        axes[0].plot(freq_avg, beta_vals[:, 0], '.r', markersize=8, label='first')
        axes[0].plot(freq_avg, beta_vals[:, 1], '.g', markersize=8, label='sec')
        axes[0].plot(freq_avg, beta_avg, '.b', markersize=10, label='mean')
        axes[0].set_ylabel('Beta')
        axes[0].set_xlabel('TM$_{010}$ Frequency [GHz]')
        axes[0].set_title('Coupling Factor')
        axes[0].legend()
        self.add_grid(axes[0])
        
        # Panel 2: Unloaded Q from beta and transmission Q
        if hasattr(scaleinfo, 'txparams'):
            unloaded_Q = (1 + beta_avg) * scaleinfo.txparams[:, 2]
            axes[1].plot(freq_avg, unloaded_Q, '.', markersize=12)
            axes[1].set_ylabel('Unloaded Q')
            axes[1].set_xlabel('TM$_{010}$ Frequency [GHz]')
            axes[1].set_title('Unloaded Q')
            self.add_grid(axes[1])
        
        # Panel 3: Reflection dip depth
        axes[2].plot(freq_avg, rfl_fit_params[:, 0], '.', markersize=12, label='first')
        axes[2].plot(freq_avg, rfl_fit_params[:, 5], '.', markersize=12, label='second')
        axes[2].set_ylabel('RFL Dip')
        axes[2].set_xlabel('TM$_{010}$ Frequency [GHz]')
        axes[2].set_title('RFL Dip')
        axes[2].legend()
        self.add_grid(axes[2])
        
        # Panel 4: Frequency difference (tx - rfl)
        if hasattr(scaleinfo, 'txparams'):
            freq_diff = (scaleinfo.txparams[:, 1] - rfl_fit_params[:, 1]) * 1e6
            axes[3].plot(freq_avg, freq_diff, '.', markersize=12)
            axes[3].set_ylabel('f$_{tx}$ - f$_{rfl}$ [kHz]')
            axes[3].set_xlabel('TM$_{010}$ Frequency [GHz]')
            axes[3].set_title('Difference in Frequency (tx - rfl)')
            self.add_grid(axes[3])
        
        # Panel 5: Reflection baseline for JPA gain
        if hasattr(rfl_baseline_db, 'rfl_base1_db') and hasattr(rfl_baseline_db, 'rfl_base2_db'):
            axes[4].plot(freq_avg, rfl_baseline_db.rfl_base1_db, '.', markersize=8, label='first')
            axes[4].plot(freq_avg, rfl_baseline_db.rfl_base2_db, '.', markersize=8, label='second')
            axes[4].set_ylabel('RFL Baseline Fit [dB]')
            axes[4].set_xlabel('TM$_{010}$ Frequency [GHz]')
            axes[4].set_title('RFL Baseline for JPA Gain')
            axes[4].legend()
            self.add_grid(axes[4])
        
        # Hide empty subplot
        axes[5].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        save_path = Path(plot_dir) / "rfl_fit_results"
        self.save_figure(fig, save_path)
        
        return fig