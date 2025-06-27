import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
from sklearn.cluster import KMeans
import os 
from beamform import beamform_frequency_band, interpolate_to_grid

def setup_output_directories(band_names):
    """Create output directories and video writers"""
    output_dir = "beamforming_frames_multiband"
    os.makedirs(output_dir, exist_ok=True)

    band_dirs = []
    video_writers = []

    for band_name in band_names:
        band_dir = os.path.join(output_dir, band_name)
        frames_dir = os.path.join(band_dir, "frames")
        clusters_dir = os.path.join(band_dir, "clusters")
        
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(clusters_dir, exist_ok=True)

        band_dirs.append({
            "root": band_dir,
            "frames": frames_dir,
            "clusters": clusters_dir
        })

        video_path = os.path.join(band_dir, f'beamforming_360_{band_name}.mp4')
        writer = imageio.get_writer(video_path, fps=10, quality=8)
        video_writers.append(writer)

    return output_dir, band_dirs, video_writers

def perform_clustering_analysis(azimuths, elevations, power_db):
    """Perform K-means clustering on high-intensity points"""
    x = azimuths
    y = elevations
    z = power_db
    threshold = np.percentile(z, 90)  # 90th percentile
    mask = z >= threshold

    x_top = x[mask]
    y_top = y[mask]
    z_top = z[mask]

    # Stack x and y for clustering
    xy_top = np.vstack((x_top, y_top)).T

    # KMeans clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=0)
    labels = kmeans.fit_predict(xy_top)
    
    return x_top, y_top, labels, kmeans.cluster_centers_

def plot_clustering_results(x_top, y_top, labels, centroids, save_path=None):
    """Plot K-means clustering results"""
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(x_top, y_top, c=labels, cmap='tab10', s=100, edgecolor='k')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

    plt.xlabel('Azimuth (°)')
    plt.ylabel('Elevation (°)')
    plt.title('K-means Clustering on Top 10% Power Points')
    plt.xlim([0, 360])
    plt.ylim([-90, 90])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        #print(f"Saved clustering plot: {save_path}")
    else:
        plt.show()

    plt.close()

def create_2d_panoramic_plot(ax, power_grid, vmin, vmax, band_name, t, max_frames, nperseg, noverlap, fs):
    """Create 2D equirectangular projection plot"""
    extent = [0, 360, -90, 90]
    im = ax.imshow(power_grid,
                   extent=extent,
                   origin='lower',
                   aspect='auto',
                   cmap='hot',
                   vmin=vmin,
                   vmax=vmax,
                   interpolation='bilinear')

    # Styling for 2D plot
    ax.set_facecolor('black')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Beamformed Power (dB)', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # Add grid
    ax.grid(True, alpha=0.3, color='white', linestyle='--', linewidth=0.5)

    # Add direction markers
    directions = {'N': 0, 'E': 90, 'S': 180, 'W': 270}
    for label, az in directions.items():
        ax.axvline(x=az, color='cyan', alpha=0.7, linestyle=':', linewidth=1)
        ax.text(az, 95, label, ha='center', va='center',
                color='cyan', fontsize=14, weight='bold')

    # Add elevation markers
    for el in [-60, -30, 0, 30, 60]:
        ax.axhline(y=el, color='white', alpha=0.2, linestyle=':')
        if el == 0:
            ax.text(365, el, 'Horizon', ha='left', va='center',
                    color='white', fontsize=10, weight='bold')

    # Labels and title for 2D plot
    ax.set_xlabel('Azimuth (degrees)', color='white', fontsize=12)
    ax.set_ylabel('Elevation (degrees)', color='white', fontsize=12)
    ax.set_title(f'{band_name} | Time: {t*(nperseg-noverlap)/fs:.2f}s | Frame {t+1}/{max_frames}',
                 color='white', fontsize=14, weight='bold')

    # Set ticks and limits
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.tick_params(colors='white')
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    
    return im

def create_3d_spherical_plot(ax, unit_vectors, power_db, vmin, vmax):
    """Create 3D spherical scatter plot"""
    ax.set_facecolor('black')

    # Create 3D scatter plot
    scatter = ax.scatter(unit_vectors[:, 0], unit_vectors[:, 1], unit_vectors[:, 2],
                         c=power_db, s=8, cmap='hot', alpha=0.8, vmin=vmin, vmax=vmax)

    # Add a wireframe sphere for reference
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color='white', linewidth=0.5)

    # Styling for 3D plot
    ax.set_xlabel('X', color='white', fontsize=10)
    ax.set_ylabel('Y', color='white', fontsize=10)
    ax.set_zlabel('Z', color='white', fontsize=10)
    ax.set_title('3D View', color='white', fontsize=12, weight='bold')

    # Set equal aspect ratio
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])

    # Style the 3D axes
    ax.tick_params(colors='white', labelsize=8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    return scatter

def create_frequency_spectrum_plot(ax, stfts, freqs, f_low, f_high, t):
    """Create frequency spectrum plot for the current band"""
    ax.set_facecolor('black')

    # Show the frequency range for this band
    freq_mask = (freqs >= f_low) & (freqs <= f_high)
    band_freqs = freqs[freq_mask]

    if len(band_freqs) > 0:
        # Average spectrum across all mics for this time frame
        spectrum = np.mean(np.abs(stfts[:, freq_mask, t]), axis=0)
        ax.semilogy(band_freqs, spectrum, 'yellow', linewidth=2, label=f'Band Spectrum')

        # Show triangular weighting
        center_freq = (f_low + f_high) / 2.0
        weights = []
        for f in band_freqs:
            if f <= center_freq:
                weight = (f - f_low) / (center_freq - f_low) if center_freq > f_low else 1.0
            else:
                weight = (f_high - f) / (f_high - center_freq) if f_high > center_freq else 1.0
            weights.append(max(0.0, weight))

        # Normalize weights for visualization
        weights = np.array(weights)
        if np.max(spectrum) > 0:
            scaled_weights = weights * np.max(spectrum) * 0.5
            ax.plot(band_freqs, scaled_weights, 'cyan', linewidth=1, alpha=0.7, label='Triangular Weight')

    ax.set_xlabel('Frequency (Hz)', color='white', fontsize=10)
    ax.set_ylabel('Magnitude', color='white', fontsize=10)
    ax.set_title(f'Band Spectrum: {f_low:.0f}-{f_high:.0f} Hz', color='white', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, color='white')
    ax.tick_params(colors='white')

def create_frame_visualization(stfts, delays, freqs, azimuths, elevations, unit_vectors,
                             band_idx, freq_bands, band_names, band_dirs, band_dynamic_ranges,
                             t, max_frames, nperseg, noverlap, fs):
    """Create complete frame visualization with all subplots"""
    f_low, f_high = freq_bands[band_idx]
    
    # Compute power for this frequency band
    power = beamform_frequency_band(stfts, delays, freqs, t,
                                  freq_range=(f_low, f_high),
                                  weighting='triangular')

    # Convert to dB with proper floor
    power_db = 10 * np.log10(np.maximum(power, np.max(power) * 1e-6))

    # Perform clustering analysis
    x_top, y_top, labels, centroids = perform_clustering_analysis(azimuths, elevations, power_db)

    cluster_save_path = os.path.join(band_dirs[band_idx]["clusters"], f"cluster_{t:04d}.png")

    # Save it
    plot_clustering_results(x_top, y_top, labels, centroids, save_path=cluster_save_path)

    # Get dynamic range for this band
    vmin, vmax = band_dynamic_ranges[band_idx]

    # Interpolate to regular grid for visualization
    power_grid, az_grid, el_grid = interpolate_to_grid(power_db, azimuths, elevations,
                                                      grid_az_res=3, grid_el_res=3)

    # Create visualization
    fig = plt.figure(figsize=(16, 10), facecolor='black')

    # Main 2D equirectangular projection
    ax1 = plt.subplot(2, 2, (1, 3))
    create_2d_panoramic_plot(ax1, power_grid, vmin, vmax, band_names[band_idx], 
                            t, max_frames, nperseg, noverlap, fs)

    # 3D view of Fibonacci points
    ax2 = plt.subplot(2, 2, 2, projection='3d')
    create_3d_spherical_plot(ax2, unit_vectors, power_db, vmin, vmax)

    # Frequency spectrum for this band
    ax3 = plt.subplot(2, 2, 4)
    create_frequency_spectrum_plot(ax3, stfts, freqs, f_low, f_high, t)

    return fig

def create_comprehensive_analysis(stfts, delays, freqs, freq_bands, band_dynamic_ranges, 
                                azimuths, elevations, max_frames):
    """Create comprehensive multi-band analysis plot"""
    print("Creating comprehensive multi-band analysis...")
    fig, axes = plt.subplots(3, 4, figsize=(24, 18), facecolor='black')
    fig.suptitle('Multi-Band Acoustic Beamforming Analysis', color='white', fontsize=16, weight='bold')

    t_example = min(10, max_frames-1)

    for band_idx, (f_low, f_high) in enumerate(freq_bands):
        if band_idx >= 12:  # Only show first 12 bands
            break

        row = band_idx // 4
        col = band_idx % 4
        ax = axes[row, col]

        # Compute power for this band
        power = beamform_frequency_band(stfts, delays, freqs, t_example,
                                      freq_range=(f_low, f_high),
                                      weighting='triangular')
        power_db = 10 * np.log10(np.maximum(power, np.max(power) * 1e-6))

        # Interpolate to grid
        power_grid, az_grid, el_grid = interpolate_to_grid(power_db, azimuths, elevations)

        # Get dynamic range for this band
        vmin, vmax = band_dynamic_ranges[band_idx]

        # Plot
        im = ax.imshow(power_grid, extent=[0, 360, -90, 90], origin='lower',
                       aspect='auto', cmap='hot', vmin=vmin, vmax=vmax)

        ax.set_facecolor('black')
        ax.set_xlabel('Azimuth (°)', color='white', fontsize=8)
        ax.set_ylabel('Elevation (°)', color='white', fontsize=8)
        ax.set_title(f'Band {band_idx+1}: {f_low:.0f}-{f_high:.0f} Hz', color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=6)
        ax.grid(True, alpha=0.2, color='white')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors='white', labelsize=6)
        cbar.set_label('dB', color='white', fontsize=8)

    plt.tight_layout()
    plt.savefig('multiband_beamforming_analysis.png', dpi=150, bbox_inches='tight',
               facecolor='black', edgecolor='none')
    plt.close()



def print_summary_statistics(positions, num_dirs, freqs, N_MEL_BANDS, freq_bands, 
                           band_dynamic_ranges, max_frames, nperseg, noverlap, fs, 
                           output_dir, band_names):
    """Print comprehensive summary statistics"""
    print("\nSummary Statistics:")
    print(f"Total microphones: {len(positions)}")
    print(f"Total directions (Fibonacci): {num_dirs}")
    print(f"Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
    print(f"Number of frequency bands: {N_MEL_BANDS}")
    print(f"Time resolution: {(nperseg-noverlap)/fs:.3f} seconds per frame")
    print(f"Frames processed: {max_frames}")
    print(f"Video duration: {max_frames * (nperseg-noverlap)/fs:.1f} seconds")

    print("\nFrequency Bands:")
    for i, ((f_low, f_high), (vmin, vmax)) in enumerate(zip(freq_bands, band_dynamic_ranges)):
        print(f"  Band {i+1}: {f_low:.1f}-{f_high:.1f} Hz, Dynamic Range: {vmin:.1f} to {vmax:.1f} dB")

    print("Multi-band analysis saved as multiband_beamforming_analysis.png")
    print("Processing complete!")