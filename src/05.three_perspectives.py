import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap
from pathlib import Path
from typing import Optional, Dict, Tuple


def create_three_perspectives_plot(
    embeddings_file: Path,
    song_data_file: Path,
    target_audio_path: Path,
    output_file: Path,
    sample_rate: int = 22050,
    umap_params: Optional[Dict] = None,
    figsize: Tuple[int, int] = (20, 11),
    dpi: int = 300
) -> None:
    """
    Create a three-perspective visualization of a target song:
    1. Producer perspective: waveform
    2. Computer perspective: mel spectrogram
    3. Data scientist perspective: 2D embedding space
    
    Parameters
    ----------
    embeddings_file : Path
        Path to .npy file containing embeddings matrix
    song_data_file : Path
        Path to .pkl file containing song metadata DataFrame
    target_audio_path : Path
        Path to target MP3 file to analyze
    output_file : Path
        Path where to save the output PNG
    sample_rate : int, default=22050
        Audio sample rate for librosa
    umap_params : dict, optional
        UMAP parameters (n_neighbors, min_dist, random_state)
    figsize : tuple, default=(20, 11)
        Figure size (width, height) in inches
    dpi : int, default=300
        Output resolution
    """
    if umap_params is None:
        umap_params = {'n_components': 2, 'random_state': 42, 
                       'n_neighbors': 15, 'min_dist': 0.1}
    
    # Load data
    print(f"Loading data from {song_data_file}...")
    df = pd.read_pickle(song_data_file)
    embeddings = np.load(embeddings_file)
    print(f"✓ Loaded {len(df)} songs with {embeddings.shape[1]}D embeddings")
    
    # Identify name column
    possible_cols = ['title', 'song_name', 'track_name', 'name', 'filename', 'file_path']
    name_col = next((col for col in possible_cols if col in df.columns), df.columns[0])
    
    # Load and process audio
    print(f"Loading audio from {target_audio_path}...")
    y, sr = librosa.load(target_audio_path, sr=sample_rate, duration=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    duration = len(y) / sr
    print(f"✓ Loaded audio: {target_audio_path.name} ({duration:.1f}s)")
    
    # UMAP dimensionality reduction
    print("Computing UMAP projection...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)
    umap_reducer = umap.UMAP(**umap_params)
    X_umap2d = umap_reducer.fit_transform(X_scaled)
    print("✓ UMAP projection complete")
    
    # Find target song in dataset
    your_idx = None
    for idx, row in df.iterrows():
        song_name = str(row[name_col])
        if name_col == 'file_path':
            song_name = Path(song_name).stem
        if 'Nordic Stars' in song_name and 'DanyR' in song_name:
            your_idx = idx
            print(f"✓ Found target song at index {idx}")
            break
    
    if your_idx is None:
        print("⚠ Warning: Target song not found in dataset")
    
    # Create figure
    print("Creating visualization...")
    fig = plt.figure(figsize=figsize, facecolor='white')
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1], 
                         hspace=0.35, wspace=0.15,
                         left=0.05, right=0.97, top=0.88, bottom=0.06)
    
    # TOP: Waveform
    ax_top = fig.add_subplot(gs[0, :])
    times = np.arange(len(y)) / sr / 60
    ax_top.plot(times, y, color='#4ECDC4', alpha=0.8, linewidth=0.5)
    ax_top.set_title('Music Producer Perspective: The Final Waveform',
                    fontsize=14, fontweight='bold', pad=10, color='#666666')
    ax_top.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax_top.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
    ax_top.grid(True, alpha=0.3, linestyle='--')
    ax_top.set_facecolor('#F8F9FA')
    
    ax_top.text(0.98, 0.95, f'{duration:.1f}s | {sr} Hz | {len(y):,} samples',
               transform=ax_top.transAxes, fontsize=9, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.85))
    
    ax_top.text(0.5, 0.15, 
               'The final result: months of studio work synthesized into a single audio waveform.\n'
               'Every peak and valley represents musical decisions made by ear and creative intuition',
               transform=ax_top.transAxes, fontsize=11, 
               ha='center', va='center', style='italic',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFE5E5', 
                        alpha=0.9, edgecolor='#FF6B6B', linewidth=2))
    
    # BOTTOM LEFT: Mel Spectrogram
    ax_left = fig.add_subplot(gs[1, 0])
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', 
                                    y_axis='mel', ax=ax_left, cmap='magma')
    ax_left.set_title('Computer Perspective: How Machines See Sound\n(Mel Spectrogram)',
                     fontsize=12, fontweight='bold', pad=8, color='#666666')
    ax_left.set_xlabel('Time', fontsize=10, fontweight='bold')
    ax_left.set_ylabel('Frequency (Hz)', fontsize=10, fontweight='bold')
    cbar = plt.colorbar(img, ax=ax_left, format='%+2.0f dB')
    cbar.set_label('dB', rotation=0, labelpad=15, fontsize=9)
    
    ax_left.text(0.5, 0.05,
                'Machines decompose audio into frequency patterns over time:\n'
                'the raw material for AI to understand musical structure',
                transform=ax_left.transAxes, fontsize=11,
                ha='center', va='center', style='italic',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFF4E5',
                         alpha=0.9, edgecolor='#FF8C00', linewidth=2))
    
    # BOTTOM RIGHT: UMAP Embedding Space
    ax_right = fig.add_subplot(gs[1, 1])
    colors = {'TRANCE': '#FF6B6B', 'ITALODANCE': '#4ECDC4'}
    
    for genre in df['genre'].unique():
        mask = df['genre'] == genre
        ax_right.scatter(X_umap2d[mask, 0], X_umap2d[mask, 1],
                        c=colors.get(genre, '#999999'), label=genre,
                        s=100, alpha=0.6, edgecolors='white', linewidth=1)
    
    if your_idx is not None:
        ax_right.scatter(X_umap2d[your_idx, 0], X_umap2d[your_idx, 1],
                        c='gold', s=500, marker='*', edgecolors='black',
                        linewidth=2.5, label='My 2014 Remix', zorder=10)
    
    ax_right.set_title('Data Scientist Perspective\n(641D → 2D via UMAP)',
                      fontsize=12, fontweight='bold', pad=8, color='#666666')
    ax_right.set_xlabel('UMAP Dimension 1', fontsize=10, fontweight='bold')
    ax_right.set_ylabel('UMAP Dimension 2', fontsize=10, fontweight='bold')
    ax_right.legend(fontsize=9, loc='upper right', framealpha=0.95, 
                   edgecolor='black', fancybox=True, shadow=True)
    ax_right.grid(True, alpha=0.2, linestyle='--')
    ax_right.set_facecolor('#F8F9FA')
    
    ax_right.text(0.5, 0.065,
                 'Each song becomes a 641-dimensional vector capturing its sonic fingerprint.\n'
                 'Once music is numbers, anything is possible: clustering, similarity search, visualization...\n'
                 'Here: two similar genres naturally separate in space',
                 transform=ax_right.transAxes, fontsize=11,
                 ha='center', va='center', style='italic',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#E5F5FF',
                          alpha=0.9, edgecolor='#4ECDC4', linewidth=2))
    
    # Main titles
    fig.text(0.5, 0.965, 'From Music Producer to Data Scientist: Different Ways of Seeing Sound',
             fontsize=17, fontweight='bold', ha='center', va='top')
    fig.text(0.5, 0.935, '"Nordic Stars - Crying in the Rain (DanyR Remix)" | 2014 → 2025',
             fontsize=14, ha='center', va='top', style='italic', color='#555', fontweight='bold')
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved plot to: {output_file}")


if __name__ == "__main__":
    # Configure paths relative to script location
    SCRIPT_DIR = Path(__file__).parent
    BASE_DIR = SCRIPT_DIR.parent
    
    EMBEDDINGS_FILE = BASE_DIR / "data/processed/embeddings_matrix.npy"
    SONG_DATA_FILE = BASE_DIR / "data/processed/song_data.pkl"
    TARGET_AUDIO = BASE_DIR / "data/raw/ITALODANCE/Nordic Stars - Crying in the rain (DanyR Italomelodic remix).mp3"
    OUTPUT_FILE = BASE_DIR / "data/plots/three_perspectives.png"
    
    # Verify input files exist
    for file_path, name in [(EMBEDDINGS_FILE, "embeddings"), 
                             (SONG_DATA_FILE, "song data"), 
                             (TARGET_AUDIO, "target audio")]:
        if not file_path.exists():
            raise FileNotFoundError(f"{name} file not found: {file_path}")
    
    # Execute visualization
    create_three_perspectives_plot(
        embeddings_file=EMBEDDINGS_FILE,
        song_data_file=SONG_DATA_FILE,
        target_audio_path=TARGET_AUDIO,
        output_file=OUTPUT_FILE,
        sample_rate=22050,
        umap_params={'n_components': 2, 'random_state': 42, 'n_neighbors': 15, 'min_dist': 0.1},
        figsize=(20, 11),
        dpi=300
    )
    
    print("\n✓ Script completed successfully")