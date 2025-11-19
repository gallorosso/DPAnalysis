"""
JPA gain plotting utilities.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from matplotlib.animation import FFMpegWriter

from .utils import ensure_dir, save_fig

def plot_jpa_gain_profiles(jpa_results: Dict[str, Any], scaleinfo: Dict[str, Any], 
                          files: List[Path], plot_dir: Path, show: bool = False) -> None:
    """
    Create individual JPA gain profile plots for each file.
    """
    plot_dir = _ensure_path(plot_dir)
    ensure_dir(plot_dir)
    
    print("  Generating JPA gain profile plots...")
    
    # Check if FFmpeg is available
    ffmpeg_available = _check_ffmpeg_available()
    
    if ffmpeg_available:
        # Create video with FFmpeg
        _create_jpa_video(jpa_results, scaleinfo, files, plot_dir)
    else:
        # Fallback: Create individual PNG files
        _create_jpa_individual_plots(jpa_results, scaleinfo, files, plot_dir)
    
    print(f"    JPA gain plots saved to: {plot_dir}")


def _check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available on the system."""
    import subprocess
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("    Warning: FFmpeg not found. Creating individual PNG files instead of video.")
        return False


def _create_jpa_video(jpa_results: Dict[str, Any], scaleinfo: Dict[str, Any],
                     files: List[Path], plot_dir: Path) -> None:
    """Create JPA gain profile video using FFmpeg."""
    from matplotlib.animation import FFMpegWriter
    
    video_path = plot_dir / "JPAgain.mp4"
    writer = FFMpegWriter(fps=3, metadata=dict(title='JPA Gain Profiles'))
    
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    
    with writer.saving(fig, str(video_path), dpi=100):
        for i, tx2_file in enumerate(files):
            try:
                _plot_single_jpa_profile(fig, i, tx2_file, jpa_results, scaleinfo, writer)
            except Exception as e:
                print(f"    Warning: Could not plot JPA profile for {tx2_file}: {e}")
                continue
    
    plt.close(fig)


def _create_jpa_individual_plots(jpa_results: Dict[str, Any], scaleinfo: Dict[str, Any],
                               files: List[Path], plot_dir: Path) -> None:
    """Create individual PNG files for each JPA profile (fallback when no FFmpeg)."""
    individual_plot_dir = plot_dir / "jpa_individual_profiles"
    ensure_dir(individual_plot_dir)
    
    for i, tx2_file in enumerate(files):
        try:
            fig = plt.figure(figsize=(12, 8), constrained_layout=True)
            _plot_single_jpa_profile_fallback(fig, i, tx2_file, jpa_results, scaleinfo)
            
            # Save individual plot
            filename = f"jpa_profile_{i:03d}.png"
            save_fig(fig, individual_plot_dir, filename.replace('.png', ''), close=True)
            
        except Exception as e:
            print(f"    Warning: Could not create individual JPA plot for {tx2_file}: {e}")
            plt.close('all')
            continue


def _plot_single_jpa_profile_fallback(fig: plt.Figure, file_index: int, tx2_file: Path,
                                    jpa_results: Dict[str, Any], scaleinfo: Dict[str, Any]) -> None:
    """
    Plot a single JPA gain profile for individual PNG file (no video writer).
    """
    plt.figure(fig.number)
    
    # Get file base and load JPA data
    file_base = _get_file_base(tx2_file)
    jpaamp_file = Path(f"{file_base}jpaamp.mat")
    
    if not jpaamp_file.exists():
        return
    
    try:
        data = scipy.io.loadmat(str(jpaamp_file))
        data2 = {}
        jpaamp2_file = Path(f"{file_base}jpaamp2.mat")
        if jpaamp2_file.exists():
            data2 = scipy.io.loadmat(str(jpaamp2_file))
        
        # Extract data for plotting (same as original function)
        f_ghz_amp = data['f_GHz_jpaamp'].flatten()
        i_amp = data['I_jpaamp'].flatten()
        q_amp = data['Q_jpaamp'].flatten()
        mag_amp_dB = 10 * np.log10(i_amp**2 + q_amp**2)
        
        # Get fit parameters
        amp_fit_params = np.array(jpa_results['amp_gain_fit'][file_index])
        sqz_fit_params = np.array(jpa_results['sqz_gain_fit'][file_index])
        
        # Create frequency array for fit curves
        f_fit = np.linspace(f_ghz_amp.min(), f_ghz_amp.max(), 200)
        
        # Plot raw data
        plt.plot(f_ghz_amp, mag_amp_dB, '-r', linewidth=1, label='Data')
        
        # Plot fit curves if available
        if np.any(amp_fit_params != 0):
            amp_fit_dB = 10 * np.log10(_lorentzian_function(f_fit, amp_fit_params))
            plt.plot(f_fit, amp_fit_dB, '-b', linewidth=2, label='Fit')
        
        # Plot secondary data if available and has squeezer gain
        sqz_gain = jpa_results['gain2Q_sqz_dB_fit'][file_index]
        if sqz_gain > 0 and data2:
            try:
                f_ghz_amp2 = data2['f_GHz_jpaamp2'].flatten()
                i_amp2 = data2['I_jpaamp2'].flatten()
                q_amp2 = data2['Q_jpaamp2'].flatten()
                mag_amp2_dB = 10 * np.log10(i_amp2**2 + q_amp2**2)
                
                plt.plot(f_ghz_amp2, mag_amp2_dB, '-c', linewidth=1, label='Data2')
                
                # Plot squeezer data
                if 'f_GHz_jpasqz' in data:
                    f_ghz_sqz = data['f_GHz_jpasqz'].flatten()
                    i_sqz = data['I_jpasqz'].flatten()
                    q_sqz = data['Q_jpasqz'].flatten()
                    mag_sqz_dB = 10 * np.log10(i_sqz**2 + q_sqz**2)
                    plt.plot(f_ghz_sqz, mag_sqz_dB, '-m', linewidth=1, label='SQZ Data')
                
                if np.any(sqz_fit_params != 0):
                    sqz_fit_dB = 10 * np.log10(_lorentzian_function(f_fit, sqz_fit_params))
                    plt.plot(f_fit, sqz_fit_dB, '-c', linewidth=2, label='SQZ Fit')
                    
            except Exception:
                pass
        
        # Add plot decorations
        bandwidth = jpa_results['JPAbandwidth'][file_index]
        plt.xlabel('Frequency [GHz]')
        plt.ylabel('Gain [dB]')
        plt.title(f'JPA: {jpaamp_file.name}\nBandwidth: {bandwidth:.0f} Hz', 
                 fontsize=10, pad=20)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim(-30, 5)
        
    except Exception as e:
        print(f"    Could not plot JPA profile for {jpaamp_file.name}: {e}")

def plot_jpa_summary(jpa_results: Dict[str, Any], scaleinfo: Dict[str, Any], 
                    plot_dir: Path, show: bool = False) -> None:
    """
    Create JPA summary trend plots across all files.
    
    Equivalent to the summary plots in JPAgainAutorun.
    
    Args:
        jpa_results: Results from JPAGainAnalysisStage  
        scaleinfo: Global scaleinfo dictionary
        plot_dir: Directory to save plots
        show: Whether to display plots interactively
    """
    plot_dir = _ensure_path(plot_dir)
    ensure_dir(plot_dir)
    
    print("  Generating JPA summary plots...")
    
    # Extract data from results
    freq_ghz_plot = np.array(jpa_results['amp_gain_fit'])[:, 1]  # f0 from amp fits
    bandwidth = np.array(jpa_results['JPAbandwidth'])
    
    # Gain arrays
    gain2Q_amp_dB_fit_corr = np.array(jpa_results['gain2Q_amp_dB_fit_corr'])
    gain2Q_amp2_dB_fit_corr = np.array(jpa_results['gain2Q_amp2_dB_fit_corr'])
    gain1Q_amp_dB = np.array(jpa_results.get('gain1Q_amp_dB', np.zeros_like(gain2Q_amp_dB_fit_corr)))
    gain1Q_amp_dB2 = np.array(jpa_results.get('gain1Q_amp_dB2', np.zeros_like(gain2Q_amp_dB_fit_corr)))
    
    gain2Q_sqz_dB_fit_corr = np.array(jpa_results['gain2Q_sqz_dB_fit_corr'])
    gain2Q_sqz2_dB_fit_corr = np.array(jpa_results['gain2Q_sqz2_dB_fit_corr'])
    gain1Q_sqz_dB = np.array(jpa_results.get('gain1Q_sqz_dB', np.zeros_like(gain2Q_sqz_dB_fit_corr)))
    gain1Q_sqz_dB2 = np.array(jpa_results.get('gain1Q_sqz_dB2', np.zeros_like(gain2Q_sqz_dB_fit_corr)))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    
    # Subplot (1,1): AMP Gain (1Q and 2Q)
    ax = axes[0, 0]
    ax.plot(freq_ghz_plot, gain2Q_amp_dB_fit_corr, '.', markersize=8, label='2Q(1st)')
    ax.plot(freq_ghz_plot, gain2Q_amp2_dB_fit_corr, '.', markersize=8, label='2Q(2nd)')
    ax.plot(freq_ghz_plot, gain1Q_amp_dB, '.', markersize=8, label='1Q(1st)')
    ax.plot(freq_ghz_plot, gain1Q_amp_dB2, '.', markersize=8, label='1Q(2nd)')
    ax.set_ylabel('Gain [dB]')
    ax.set_xlabel(r'TM$_{010}$ Frequency [GHz]')
    ax.set_title('AMP Gain (1Q and 2Q)')
    ax.legend(ncol=2, fontsize=9)
    
    # Subplot (1,2): SQZ Gain (1Q and 2Q)
    ax = axes[0, 1]
    ax.plot(freq_ghz_plot, gain2Q_sqz_dB_fit_corr, '.', markersize=8, label='2Q(1st)')
    ax.plot(freq_ghz_plot, gain2Q_sqz2_dB_fit_corr, '.', markersize=8, label='2Q(2nd)')
    ax.plot(freq_ghz_plot, gain1Q_sqz_dB, '.', markersize=8, label='1Q(1st)')
    ax.plot(freq_ghz_plot, gain1Q_sqz_dB2, '.', markersize=8, label='1Q(2nd)')
    ax.set_ylabel('Gain [dB]')
    ax.set_xlabel(r'TM$_{010}$ Frequency [GHz]')
    ax.set_title('SQZ Gain (1Q and 2Q)')
    ax.legend(ncol=2, fontsize=9)
    
    # Subplot (1,3): JPA Bandwidth
    ax = axes[0, 2]
    ax.plot(freq_ghz_plot, bandwidth, '.', markersize=8)
    ax.set_ylabel('JPA Bandwidth (AMP) [Hz]')
    ax.set_xlabel(r'TM$_{010}$ Frequency [GHz]')
    ax.set_title('Bandwidth of amplifier JPA')
    
    # Subplot (2,1): AMP Gain Differences
    ax = axes[1, 0]
    ax.plot(freq_ghz_plot, gain1Q_amp_dB - gain1Q_amp_dB2, '.', markersize=8, label='1Q')
    ax.plot(freq_ghz_plot, gain2Q_amp_dB_fit_corr - gain2Q_amp2_dB_fit_corr, '.', markersize=8, label='2Q')
    ax.set_ylabel('Gain Change [dB]')
    ax.set_xlabel(r'TM$_{010}$ Frequency [GHz]')
    ax.set_title('AMP Diff (1st-2nd)')
    ax.legend()
    
    # Subplot (2,2): SQZ Gain Differences  
    ax = axes[1, 1]
    ax.plot(freq_ghz_plot, gain1Q_sqz_dB - gain1Q_sqz_dB2, '.', markersize=8, label='1Q')
    ax.plot(freq_ghz_plot, gain2Q_sqz_dB_fit_corr - gain2Q_sqz2_dB_fit_corr, '.', markersize=8, label='2Q')
    ax.set_ylabel('Gain Change (1st-2nd) [dB]')
    ax.set_xlabel(r'TM$_{010}$ Frequency [GHz]')
    ax.set_title('SQZ Diff (1st-2nd)')
    ax.legend()
    
    # Subplot (2,3): Difference Between 1Q and 2Q Methods
    ax = axes[1, 2]
    ax.plot(freq_ghz_plot, gain1Q_amp_dB - gain2Q_amp_dB_fit_corr, '.', markersize=8, label='AMP')
    ax.plot(freq_ghz_plot, gain1Q_sqz_dB - gain2Q_sqz_dB_fit_corr, '.', markersize=8, label='SQZ')
    ax.set_ylabel('(1Q - 2Q) [dB]')
    ax.set_xlabel(r'TM$_{010}$ Frequency [GHz]')
    ax.set_title('Difference Between 1Q and 2Q Gain')
    ax.legend()
    
    # Save summary plot
    save_fig(fig, plot_dir, "jpa_fit_parameters", close=not show)
    
    if show:
        plt.show()


def _plot_single_jpa_profile(fig: plt.Figure, file_index: int, tx2_file: Path,
                           jpa_results: Dict[str, Any], scaleinfo: Dict[str, Any],
                           writer: FFMpegWriter) -> None:
    """
    Plot a single JPA gain profile for video frame.
    """
    plt.figure(fig.number)
    plt.clf()
    
    # Get file base and load JPA data
    file_base = _get_file_base(tx2_file)
    jpaamp_file = Path(f"{file_base}jpaamp.mat")
    
    if not jpaamp_file.exists():
        return
    
    try:
        data = scipy.io.loadmat(str(jpaamp_file))
        data2 = {}
        jpaamp2_file = Path(f"{file_base}jpaamp2.mat")
        if jpaamp2_file.exists():
            data2 = scipy.io.loadmat(str(jpaamp2_file))
        
        # Extract data for plotting
        f_ghz_amp = data['f_GHz_jpaamp'].flatten()
        i_amp = data['I_jpaamp'].flatten()
        q_amp = data['Q_jpaamp'].flatten()
        mag_amp_dB = 10 * np.log10(i_amp**2 + q_amp**2)
        
        # Get fit parameters
        amp_fit_params = np.array(jpa_results['amp_gain_fit'][file_index])
        sqz_fit_params = np.array(jpa_results['sqz_gain_fit'][file_index])
        
        # Create frequency array for fit curves
        f_fit = np.linspace(f_ghz_amp.min(), f_ghz_amp.max(), 200)
        
        # Plot raw data
        plt.plot(f_ghz_amp, mag_amp_dB, '-r', linewidth=1, label='Data')
        
        # Plot fit curves if available
        if np.any(amp_fit_params != 0):
            amp_fit_dB = 10 * np.log10(_lorentzian_function(f_fit, amp_fit_params))
            plt.plot(f_fit, amp_fit_dB, '-b', linewidth=2, label='Fit')
        
        # Plot secondary data if available and has squeezer gain
        sqz_gain = jpa_results['gain2Q_sqz_dB_fit'][file_index]
        if sqz_gain > 0 and data2:
            try:
                f_ghz_amp2 = data2['f_GHz_jpaamp2'].flatten()
                i_amp2 = data2['I_jpaamp2'].flatten()
                q_amp2 = data2['Q_jpaamp2'].flatten()
                mag_amp2_dB = 10 * np.log10(i_amp2**2 + q_amp2**2)
                
                plt.plot(f_ghz_amp2, mag_amp2_dB, '-c', linewidth=1, label='Data2')
                
                # Plot squeezer data
                if 'f_GHz_jpasqz' in data:
                    f_ghz_sqz = data['f_GHz_jpasqz'].flatten()
                    i_sqz = data['I_jpasqz'].flatten()
                    q_sqz = data['Q_jpasqz'].flatten()
                    mag_sqz_dB = 10 * np.log10(i_sqz**2 + q_sqz**2)
                    plt.plot(f_ghz_sqz, mag_sqz_dB, '-m', linewidth=1, label='SQZ Data')
                
                if np.any(sqz_fit_params != 0):
                    sqz_fit_dB = 10 * np.log10(_lorentzian_function(f_fit, sqz_fit_params))
                    plt.plot(f_fit, sqz_fit_dB, '-c', linewidth=2, label='SQZ Fit')
                    
            except Exception:
                pass
        
        # Add plot decorations
        bandwidth = jpa_results['JPAbandwidth'][file_index]
        plt.xlabel('Frequency [GHz]')
        plt.ylabel('Gain [dB]')
        plt.title(f'JPA: {jpaamp_file.name}\nBandwidth: {bandwidth:.0f} Hz', 
                 fontsize=10, pad=20)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # Set consistent Y-axis limits
        plt.ylim(-30, 5)
        
        # Write frame to video
        writer.grab_frame()
        
    except Exception as e:
        print(f"    Could not plot JPA profile for {jpaamp_file.name}: {e}")


def _get_file_base(tx2_file: Path) -> Path:
    """Extract file base name from tx2 file path."""
    filename = tx2_file.name.replace('tx2.mat', '')
    return tx2_file.parent / filename


def _lorentzian_function(f: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Evaluate Lorentzian + linear function."""
    if len(params) < 5 or np.all(params == 0):
        return np.zeros_like(f)
    
    P_max, f0, Q, slope, offset = params
    
    # Handle division by zero
    if Q == 0:
        return np.zeros_like(f)
    
    # Lorentzian term
    lorentzian = P_max / (1 + 4 * Q**2 * ((f / f0) - 1)**2)
    
    # Linear baseline
    linear = slope * f + offset
    
    return lorentzian + linear

def _ensure_path(path) -> Path:
    """Ensure path is Path object."""
    return path if isinstance(path, Path) else Path(path)