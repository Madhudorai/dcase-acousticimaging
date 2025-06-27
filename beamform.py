import numpy as np
from scipy.interpolate import griddata

def fibonacci_sphere(n_points=1000):
    """
    Generate points on a sphere using Fibonacci tessellation
    This provides a nearly uniform distribution of points
    """
    golden_ratio = (1 + 5**0.5) / 2  # Golden ratio

    indices = np.arange(0, n_points, dtype=float) + 0.5

    # Fibonacci spiral on sphere
    theta = np.arccos(1 - 2 * indices / n_points)  # Colatitude
    phi = np.pi * (1 + golden_ratio) * indices     # Azimuth

    # Convert to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    unit_vectors = np.column_stack((x, y, z))

    # Convert back to spherical for reference
    elevations = 90 - np.rad2deg(theta)  # Convert colatitude to elevation
    azimuths = np.rad2deg(phi) % 360     # Ensure azimuth is in [0, 360)

    return unit_vectors, azimuths, elevations


def beamform_frequency_band(stfts, delays, freqs, frame_idx, freq_range,
                           weighting='triangular', center_freq=None):
    """
    Compute beamformed power for all directions at a specific time frame
    for a specific frequency band

    Parameters:
    -----------
    stfts : ndarray
        STFT data [mics, freqs, frames]
    delays : ndarray
        Time delays for steering vectors [directions, mics]
    freqs : ndarray
        Frequency bins
    frame_idx : int
        Time frame index
    freq_range : tuple
        Frequency range (min_freq, max_freq) in Hz
    weighting : str
        'uniform' for uniform weighting, 'triangular' for triangular weighting
    center_freq : float or None
        Center frequency for triangular weighting. If None, uses middle of freq_range

    Returns:
    --------
    power : ndarray
        Beamformed power for each direction
    """
    power = np.zeros(delays.shape[0])

    # Find frequency range indices
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    active_freqs = np.where(freq_mask)[0]

    if len(active_freqs) == 0:
        return power

    # Set center frequency for triangular weighting
    if center_freq is None:
        center_freq = (freq_range[0] + freq_range[1]) / 2.0

    for bin_idx in active_freqs:
        f = freqs[bin_idx]

        # Calculate steering vector for this frequency
        phase_shifts = 2 * np.pi * f * delays
        steering_vectors = np.exp(-1j * phase_shifts)

        # Get STFT values for this bin and frame
        X = stfts[:, bin_idx, frame_idx]

        # Compute beamformed output with proper normalization
        beamformed = np.abs(np.sum(steering_vectors * X[np.newaxis, :], axis=1)) ** 2

        # Apply frequency weighting
        if weighting == 'uniform':
            weight = 1.0
        elif weighting == 'triangular':
            # Triangular weighting centered at center_freq
            if f <= center_freq:
                # Rising edge: from freq_range[0] to center_freq
                weight = (f - freq_range[0]) / (center_freq - freq_range[0]) if center_freq > freq_range[0] else 1.0
            else:
                # Falling edge: from center_freq to freq_range[1]
                weight = (freq_range[1] - f) / (freq_range[1] - center_freq) if freq_range[1] > center_freq else 1.0
            # Ensure weight doesn't go below 0
            weight = max(0.0, weight)
        else:
            raise ValueError(f"Unknown weighting type: {weighting}")

        power += beamformed * weight

    return power

def interpolate_to_grid(power_values, azimuths, elevations, grid_az_res=2, grid_el_res=2):
    """
    Interpolate scattered Fibonacci points to a regular grid for visualization
    """
    # Create regular grid
    grid_az = np.arange(0, 360, grid_az_res)
    grid_el = np.arange(-90, 90 + grid_el_res, grid_el_res)
    grid_az_mesh, grid_el_mesh = np.meshgrid(grid_az, grid_el)

    # Points for interpolation
    points = np.column_stack((azimuths, elevations))
    grid_points = np.column_stack((grid_az_mesh.flatten(), grid_el_mesh.flatten()))

    # Handle azimuth wraparound by creating duplicates
    az_wrapped = azimuths.copy()
    el_wrapped = elevations.copy()
    power_wrapped = power_values.copy()

    # Add points near boundaries to handle wraparound
    mask_near_0 = azimuths < 20
    mask_near_360 = azimuths > 340

    if np.any(mask_near_0):
        az_wrapped = np.concatenate([az_wrapped, azimuths[mask_near_0] + 360])
        el_wrapped = np.concatenate([el_wrapped, elevations[mask_near_0]])
        power_wrapped = np.concatenate([power_wrapped, power_values[mask_near_0]])

    if np.any(mask_near_360):
        az_wrapped = np.concatenate([az_wrapped, azimuths[mask_near_360] - 360])
        el_wrapped = np.concatenate([el_wrapped, elevations[mask_near_360]])
        power_wrapped = np.concatenate([power_wrapped, power_values[mask_near_360]])

    points_wrapped = np.column_stack((az_wrapped, el_wrapped))

    # Interpolate using griddata
    try:
        grid_values = griddata(points_wrapped, power_wrapped, grid_points,
                              method='cubic', fill_value=np.nan)
        grid_values = grid_values.reshape(grid_az_mesh.shape)

        # Fill any remaining NaN values with nearest neighbor
        if np.any(np.isnan(grid_values)):
            grid_values_nn = griddata(points_wrapped, power_wrapped, grid_points,
                                    method='nearest')
            grid_values_nn = grid_values_nn.reshape(grid_az_mesh.shape)
            mask = np.isnan(grid_values)
            grid_values[mask] = grid_values_nn[mask]

    except Exception as e:
        print(f"Cubic interpolation failed: {e}, falling back to linear")
        grid_values = griddata(points_wrapped, power_wrapped, grid_points,
                              method='linear', fill_value=np.nan)
        grid_values = grid_values.reshape(grid_az_mesh.shape)

        # Fill NaN with nearest neighbor
        if np.any(np.isnan(grid_values)):
            grid_values_nn = griddata(points_wrapped, power_wrapped, grid_points,
                                    method='nearest')
            grid_values_nn = grid_values_nn.reshape(grid_az_mesh.shape)
            mask = np.isnan(grid_values)
            grid_values[mask] = grid_values_nn[mask]

    return grid_values, grid_az_mesh, grid_el_mesh

def setup_beamforming_directions(n_points=1024):
    """Setup beamforming directions using Fibonacci tessellation"""
    unit_vectors, azimuths, elevations = fibonacci_sphere(n_points=n_points)
    num_dirs = len(unit_vectors)
    print(f"Created {num_dirs} directions using Fibonacci tessellation")
    return unit_vectors, azimuths, elevations, num_dirs

def compute_time_delays(positions, unit_vectors, c=343.0):
    """Precompute time delays for all directions and microphones"""
    num_dirs = len(unit_vectors)
    print(f"Precomputing delays for {num_dirs} directions...")
    delays = np.zeros((num_dirs, 32))
    for d in range(num_dirs):
        for m in range(32):
            # Time delay = (r_m · d) / c where r_m is mic position and d is direction
            delays[d, m] = np.dot(positions[m], unit_vectors[d]) / c

    print(f"Delay range: {delays.min():.6f} to {delays.max():.6f} seconds")
    return delays
