import os 
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

from tools import load_and_process_audio, setup_frequency_bands, compute_dynamic_ranges 
from plot_tools import setup_output_directories, create_frame_visualization, create_comprehensive_analysis, print_summary_statistics
from microphone import setup_microphone_array
from beamform import setup_beamforming_directions, compute_time_delays

def main(file_path,N_MEL_BANDS):
    # Constants
    c = 343.0  # Speed of sound (m/s)
    fs = 48000  # Sampling rate (Hz)

    # Setup microphone array
    positions = setup_microphone_array()
    
    # Setup beamforming directions
    unit_vectors, azimuths, elevations, num_dirs = setup_beamforming_directions()
    
    # Compute time delays
    delays = compute_time_delays(positions, unit_vectors, c)
    
    # Load and process audio
    stfts, freqs, num_frames, nperseg, noverlap = load_and_process_audio(file_path)
    
    # Setup frequency bands
    freq_bands, band_names = setup_frequency_bands(fs, 2048, N_MEL_BANDS)
    
    # Compute dynamic ranges
    band_dynamic_ranges = compute_dynamic_ranges(stfts, delays, freqs, freq_bands, num_frames)
    
    # Setup output directories
    output_dir, band_dirs, video_writers = setup_output_directories(band_names)
    
    # Process frames for each frequency band independently
    max_frames = num_frames  # Limit for demonstration
    print(f"Processing {max_frames} frames for {N_MEL_BANDS} frequency bands...")

    for t in tqdm(range(max_frames), desc="Processing frames"):
        # Process each frequency band independently
        for band_idx in range(N_MEL_BANDS):
            # Create frame visualization
            fig = create_frame_visualization(stfts, delays, freqs, azimuths, elevations, unit_vectors,
                                        band_idx, freq_bands, band_names,band_dirs, band_dynamic_ranges,
                                        t, max_frames, nperseg, noverlap, fs)

            # Save frame for this frequency band
            frame_path = os.path.join(band_dirs[band_idx]["frames"], f'frame_{t:04d}.png')
            plt.savefig(frame_path, bbox_inches='tight', dpi=100,
                    facecolor='black', edgecolor='none')
            plt.close(fig)

            # Add to video for this band
            try:
                frame_img = imageio.imread(frame_path)
                video_writers[band_idx].append_data(frame_img)
            except Exception as e:
                print(f"Error adding frame {t} for band {band_idx}: {e}")

    # Close all video writers
    for writer in video_writers:
        writer.close()

    print(f"Beamforming videos saved for {N_MEL_BANDS} frequency bands")

    # Create comprehensive analysis
    create_comprehensive_analysis(stfts, delays, freqs, freq_bands, band_dynamic_ranges, 
                                azimuths, elevations, max_frames)
    
    # Print summary statistics
    print_summary_statistics(positions, num_dirs, freqs, N_MEL_BANDS, freq_bands, 
                           band_dynamic_ranges, max_frames, nperseg, noverlap, fs, 
                           output_dir, band_names)

if __name__ == "__main__":
    file_path = "path/to/audio_array_eigenmike.wav"
    N_MEL_BANDS=12
    main(file_path, N_MEL_BANDS)