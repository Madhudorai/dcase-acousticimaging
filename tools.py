import numpy as np
import soundfile as sf
from scipy.signal import stft
from tqdm import tqdm
from scipy.interpolate import griddata
import librosa
from beamform import beamform_frequency_band
def load_and_process_audio(audio_file, fs=48000):
    """Load audio data and compute STFT for all channels"""
    # Load audio data
    signals, sample_rate = sf.read(audio_file)
    signals = signals.T
    min_length = signals.shape[1]

    print(f"Loaded 32 channels, {min_length} samples each")

    # STFT parameters
    nperseg = 2048   # Larger window for better frequency resolution
    noverlap = 1024  # 50% overlap
    nfft = 2048      # Match window size
    freqs = np.fft.rfftfreq(nfft, 1/fs)  # Frequency bins

    # Compute STFT for all channels
    print("Computing STFTs...")
    stfts = []
    for i in range(32):
        _, _, Zxx = stft(signals[i], fs=fs, nperseg=nperseg,
                        noverlap=noverlap, nfft=nfft)
        stfts.append(Zxx)

    stfts = np.array(stfts)  # Shape: (32, freq_bins, time_frames)
    num_bins = stfts.shape[1]
    num_frames = stfts.shape[2]
    print(f"STFT computed: {num_bins} freq bins, {num_frames} time frames")
    
    return stfts, freqs, num_frames, nperseg, noverlap

def setup_frequency_bands(fs=48000, nfft=2048, N_MEL_BANDS=12):
    """Setup Mel frequency bands for multi-band analysis"""
    # Generate Mel filterbank and frequency bands
    mel_filters = librosa.filters.mel(sr=fs, n_fft=nfft, n_mels=N_MEL_BANDS,
                                    fmin=100, fmax=8000)

    # Convert mel filter frequencies to actual frequency ranges
    mel_freqs = librosa.mel_frequencies(n_mels=N_MEL_BANDS+2, fmin=100, fmax=8000)

    # Create frequency band ranges from mel frequencies
    freq_bands = []
    band_names = []
    for i in range(N_MEL_BANDS):
        freq_bands.append((mel_freqs[i], mel_freqs[i+2]))  # Overlapping bands
        band_names.append(f"Band_{i+1:02d}_{mel_freqs[i]:.0f}-{mel_freqs[i+2]:.0f}Hz")

    print(f"Using {N_MEL_BANDS} Mel frequency bands:")
    for i, (f_low, f_high) in enumerate(freq_bands):
        print(f"  Band {i+1}: {f_low:.1f} - {f_high:.1f} Hz")
    
    return freq_bands, band_names

def compute_dynamic_ranges(stfts, delays, freqs, freq_bands, num_frames):
    """Compute dynamic range for each frequency band"""
    print("Computing dynamic range for each frequency band...")
    band_dynamic_ranges = []

    for band_idx, (f_low, f_high) in enumerate(freq_bands):
        print(f"Computing dynamic range for Band {band_idx+1}: {f_low:.1f}-{f_high:.1f} Hz")

        sample_powers = []
        # Sample a subset of frames to compute dynamic range
        sample_frames = range(0, num_frames, 5)  # Sample every 5th frame, up to 50 frames

        for t in sample_frames:
            power = beamform_frequency_band(stfts, delays, freqs, t,
                                        freq_range=(f_low, f_high),
                                        weighting='triangular')
            # Convert to dB with proper floor
            power_db = 10 * np.log10(np.maximum(power, np.max(power) * 1e-6))
            sample_powers.extend(power_db[np.isfinite(power_db)])

        if len(sample_powers) > 0:
            sample_powers = np.array(sample_powers)
            vmin = np.percentile(sample_powers, 1)
            vmax = np.percentile(sample_powers, 99)
            band_dynamic_ranges.append((vmin, vmax))
            print(f"  Dynamic range: {vmin:.1f} to {vmax:.1f} dB")
        else:
            # Fallback range
            band_dynamic_ranges.append((-60, 0))
            print(f"  Using fallback dynamic range: -60 to 0 dB")
    
    return band_dynamic_ranges

