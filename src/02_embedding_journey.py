"""
The Embedding Journey: From Audio Waves to Mathematical Vectors.

Creates a visual narrative showing how raw audio transforms into 
embeddings that capture musical essence. Feature extraction aligned
with the production pipeline (641 dimensions).
"""
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import EMBEDDINGS_FILE, SONG_DATA_FILE, PLOTS_DIR, HTML_DIR, SAMPLE_RATE, N_MFCC
except ModuleNotFoundError:
    BASE_DIR = Path(__file__).parent.parent
    EMBEDDINGS_FILE = BASE_DIR / "data" / "processed" / "embeddings_matrix.npy"
    SONG_DATA_FILE = BASE_DIR / "data" / "processed" / "song_data.pkl"
    PLOTS_DIR = BASE_DIR / "data" / "plots"
    HTML_DIR = BASE_DIR / "data" / "html"
    SAMPLE_RATE = 22050
    N_MFCC = 20


class EmbeddingJourneyVisualizer:
    """
    Visualize the transformation from audio to embeddings.
    
    Shows the complete pipeline: raw audio → features → embeddings → clusters.
    Feature extraction matches production pipeline exactly.
    """
    
    def __init__(self):
        """Initialize with dataset and validate dimensions."""
        self.df = pd.read_pickle(SONG_DATA_FILE)
        self.embeddings = np.load(EMBEDDINGS_FILE)
        
        # Use actual dimension from saved embeddings
        self.expected_dim = self.embeddings.shape[1]
        actual_dim = self.embeddings.shape[1]
        
        print(f"   Using {actual_dim}D embeddings from production pipeline")
        print(f"   Available columns: {list(self.df.columns)}")
        
        # Identify song name column
        possible_name_cols = ['title', 'song_name', 'track_name', 'name', 'filename', 'file_path']
        self.name_col = None
        for col in possible_name_cols:
            if col in self.df.columns:
                self.name_col = col
                break
        
        if self.name_col is None:
            # Use first string column or index
            string_cols = self.df.select_dtypes(include=['object']).columns
            if len(string_cols) > 0:
                self.name_col = string_cols[0]
            else:
                self.df['song_id'] = [f"Song {i+1}" for i in range(len(self.df))]
                self.name_col = 'song_id'
        
        print(f"   Using '{self.name_col}' for song names")
        print(f"✓ Loaded {len(self.df)} songs with {actual_dim}-dimensional embeddings")
    
    def extract_complete_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract complete feature vector matching production pipeline.
        
        Replicates extract_audio_features.py logic exactly to ensure
        consistency between extraction and visualization.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Complete 641-dimensional feature vector
        """
        # 1. Timbral features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # 2. Harmonic features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # 3. Spectral features
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spec_flatness = librosa.feature.spectral_flatness(y=y)
        
        # 4. Rhythmic features
        zcr = librosa.feature.zero_crossing_rate(y)
        
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if np.isnan(tempo) or tempo == 0:
                tempo = 120.0
        except Exception:
            tempo = 120.0
        
        # 5. Energy features
        rms = librosa.feature.rms(y=y)
        
        # 6. Tonnetz
        try:
            harmonic = librosa.effects.harmonic(y)
            if len(harmonic) > 0:
                tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
            else:
                tonnetz = np.zeros((6, 1))
        except Exception:
            tonnetz = np.zeros((6, 1))
        
        # 7. Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Aggregate to fixed-length vector (same as production)
        features = np.concatenate([
            np.mean(mfcc, axis=1),              # 20
            np.std(mfcc, axis=1),               # 20
            np.mean(mfcc_delta, axis=1),        # 20
            np.mean(chroma, axis=1),            # 12
            np.std(chroma, axis=1),             # 12
            [np.mean(spec_centroid)],           # 1
            [np.std(spec_centroid)],            # 1
            [np.mean(spec_rolloff)],            # 1
            [np.std(spec_rolloff)],             # 1
            [np.mean(spec_bandwidth)],          # 1
            [np.std(spec_bandwidth)],           # 1
            np.mean(spec_contrast, axis=1),     # 7
            np.std(spec_contrast, axis=1),      # 7
            [np.mean(spec_flatness)],           # 1
            [np.mean(zcr)],                     # 1
            [np.std(zcr)],                      # 1
            [tempo],                            # 1
            [np.mean(rms)],                     # 1
            [np.std(rms)],                      # 1
            np.mean(tonnetz, axis=1),           # 6
            np.std(tonnetz, axis=1),            # 6
            np.mean(mel_spec_db, axis=1),       # 128
            np.percentile(mel_spec_db, 25, axis=1),  # 128
            np.percentile(mel_spec_db, 50, axis=1),  # 128
            np.percentile(mel_spec_db, 75, axis=1),  # 128
        ])
        
        return features.flatten()
    
    def _get_song_names(self, mask):
        """Extract song names, handling file paths if necessary."""
        names = self.df[mask][self.name_col].values
        
        # If using file_path, extract just the filename
        if self.name_col == 'file_path':
            names = [Path(p).stem for p in names]
        
        return names
    
    def visualize_single_song_journey(
        self, 
        song_path: str,
        duration: Optional[float] = None,
        save: bool = True
    ):
        """
        Complete visualization of one song's transformation.
        
        Shows: Waveform → Spectrogram → Key Features → Complete Embedding
        
        Args:
            song_path: Path to audio file
            duration: Duration to analyze (None = entire song)
            save: Whether to save plot
        """
        song_name = Path(song_path).stem
        
        y, sr = librosa.load(song_path, sr=SAMPLE_RATE, duration=duration)
        actual_duration = len(y) / sr
        
        print(f"   Analyzing {actual_duration:.1f} seconds of audio...")
        
        # Extract complete features matching production pipeline
        complete_features = self.extract_complete_features(y, sr)
        actual_dim = len(complete_features)
        
        print(f"   Complete embedding: {actual_dim} dimensions")
        
        # Also extract individual features for visualization
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        fig = plt.figure(figsize=(22, 15))
        
        fig.text(0.5, 0.985, f'The Embedding Journey: {song_name}', 
                fontsize=18, fontweight='bold', ha='center')
        
        gs = fig.add_gridspec(4, 3, hspace=0.38, wspace=0.25,
                             left=0.05, right=0.98, top=0.955, bottom=0.04)
        
        # Step 1: Waveform
        ax1 = fig.add_subplot(gs[0, :])
        librosa.display.waveshow(y, sr=sr, ax=ax1, color='#4ECDC4', alpha=0.8)
        ax1.set_title('STEP 1: Raw Audio Waveform (Time Domain)', 
                     fontsize=12, fontweight='bold', pad=10)
        ax1.set_xlabel('Time (s)', fontsize=10)
        ax1.set_ylabel('Amplitude', fontsize=10)
        ax1.text(0.015, 0.95, f'Input: {len(y):,} samples @ {sr} Hz ({actual_duration:.1f}s)', 
                transform=ax1.transAxes, fontsize=9, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax1.grid(True, alpha=0.3)
        
        ax1.text(0.985, 0.05, 
                'Raw digital audio: amplitude over time',
                transform=ax1.transAxes, fontsize=8, 
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         alpha=0.8, edgecolor='gray', linewidth=1))
        
        # Step 2: Mel Spectrogram
        ax2 = fig.add_subplot(gs[1, :])
        img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', 
                                       y_axis='mel', ax=ax2, cmap='magma')
        ax2.set_title('STEP 2: Mel Spectrogram (Frequency Domain)', 
                     fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel('Time', fontsize=10)
        ax2.set_ylabel('Hz', fontsize=10)
        cbar2 = fig.colorbar(img, ax=ax2, format='%+2.0f dB')
        cbar2.set_label('dB', rotation=0, labelpad=15, fontsize=9)
        ax2.text(0.015, 0.95, f'Transform: {mel_spec.shape[0]} freq bins x {mel_spec.shape[1]} time frames', 
                transform=ax2.transAxes, fontsize=9, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.text(0.985, 0.05,
                'Frequency-time decomposition\nBrighter = more energy',
                transform=ax2.transAxes, fontsize=8,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         alpha=0.8, edgecolor='gray', linewidth=1))
        
        # Step 3a: MFCC
        ax3a = fig.add_subplot(gs[2, 0])
        img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax3a, cmap='RdBu_r')
        ax3a.set_title('STEP 3a: MFCC\n(Timbre)', fontsize=10, fontweight='bold', pad=8)
        ax3a.set_ylabel('Coefficient', fontsize=9)
        ax3a.set_xlabel('Time', fontsize=9)
        fig.colorbar(img, ax=ax3a)
        
        ax3a.text(0.97, 0.03,
                'Sound texture:\ndistinguishes instruments',
                transform=ax3a.transAxes, fontsize=7.5,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         alpha=0.85, edgecolor='gray', linewidth=0.8))
        
        # Step 3b: Chroma
        ax3b = fig.add_subplot(gs[2, 1])
        img = librosa.display.specshow(chroma, sr=sr, x_axis='time', 
                                       y_axis='chroma', ax=ax3b, cmap='coolwarm')
        ax3b.set_title('STEP 3b: Chroma\n(Harmony)', fontsize=10, fontweight='bold', pad=8)
        ax3b.set_xlabel('Time', fontsize=9)
        fig.colorbar(img, ax=ax3b)
        
        ax3b.text(0.97, 0.03,
                '12 musical notes:\nchord structure',
                transform=ax3b.transAxes, fontsize=7.5,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         alpha=0.85, edgecolor='gray', linewidth=0.8))
        
        # Step 3c: Spectral Centroid
        ax3c = fig.add_subplot(gs[2, 2])
        frames = range(len(spec_centroid))
        t = librosa.frames_to_time(frames, sr=sr)
        ax3c.plot(t, spec_centroid, linewidth=2, color='#FF6B6B')
        ax3c.fill_between(t, 0, spec_centroid, alpha=0.3, color='#FF6B6B')
        ax3c.set_title('STEP 3c: Spectral Centroid\n(Brightness)', 
                      fontsize=10, fontweight='bold', pad=8)
        ax3c.set_xlabel('Time (s)', fontsize=9)
        ax3c.set_ylabel('Hz', fontsize=9)
        ax3c.grid(True, alpha=0.3)
        
        ax3c.text(0.97, 0.97,
                'Sound brightness:\nhigh = sharp, low = warm',
                transform=ax3c.transAxes, fontsize=7.5,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         alpha=0.85, edgecolor='gray', linewidth=0.8))
        
        # Step 4: Complete embedding vector
        ax4 = fig.add_subplot(gs[3, :])
        
        # Normalize for visualization
        p5 = np.percentile(complete_features, 5)
        p95 = np.percentile(complete_features, 95)
        features_clipped = np.clip(complete_features, p5, p95)
        features_normalized = (features_clipped - p5) / (p95 - p5 + 1e-8)
        
        # Multi-row display
        n_rows = 5
        dims_per_row = int(np.ceil(actual_dim / n_rows))
        
        features_display = np.full((n_rows, dims_per_row), np.nan)
        
        for i, val in enumerate(features_normalized):
            row = i // dims_per_row
            col = i % dims_per_row
            if row < n_rows:
                features_display[row, col] = val
        
        # Heatmap
        im = ax4.imshow(features_display, cmap='RdYlBu_r', aspect='auto', 
                       interpolation='nearest', vmin=0, vmax=1)
        
        ax4.set_title(f'STEP 4: Complete Production Embedding Vector ({actual_dim} dimensions)\n'
                     f'Each cell = one feature | Normalized [0-1] | Matches saved embeddings', 
                     fontsize=12, fontweight='bold', pad=12)
        ax4.set_ylabel('Row', fontsize=10)
        ax4.set_xlabel('Dimension Index', fontsize=10)
        ax4.set_yticks(range(n_rows))
        ax4.set_yticklabels([f'Row {i+1}' for i in range(n_rows)], fontsize=9)
        
        # Feature group boundaries (correct for 641D)
        feature_boundaries = [
            (0, 'MFCC\nmean'),
            (20, 'MFCC\nstd'),
            (40, 'MFCC\nΔ'),
            (60, 'Chroma\nmean'),
            (72, 'Chroma\nstd'),
            (84, 'Spectral\nfeats'),
            (103, 'Rhythm'),
            (106, 'Energy'),
            (108, 'Tonnetz'),
            (120, 'Mel Spectrum\n(128 × 4 stats)')
        ]
        
        for boundary_idx, label in feature_boundaries:
            if boundary_idx < actual_dim:
                row = boundary_idx // dims_per_row
                col = boundary_idx % dims_per_row
                
                if row < n_rows:
                    ax4.axvline(col - 0.5, color='white', linestyle='--', 
                               alpha=0.8, linewidth=1.5)
                    
                    if col < dims_per_row - 5:
                        ax4.text(col + 2, row, label, 
                                color='white', fontweight='bold', 
                                fontsize=8, va='center',
                                bbox=dict(boxstyle='round', facecolor='black', 
                                        alpha=0.7, edgecolor='white', linewidth=1))
        
        cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Value', rotation=270, labelpad=18, fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        
        if save:
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = PLOTS_DIR / f"embedding_journey_{song_name[:30]}.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Journey visualization saved: {filepath}")
        
        plt.close()
        
        return complete_features
    
    def visualize_embedding_space_2d(self, save: bool = True):
        """
        2D UMAP projection with song titles on hover.
        """
        from sklearn.preprocessing import StandardScaler
        import umap
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.embeddings)
        
        umap_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap2d = umap_2d.fit_transform(X_scaled)
        
        colors = {'TRANCE': '#FF6B6B', 'ITALODANCE': '#4ECDC4'}
        
        fig = go.Figure()
        
        for genre in self.df['genre'].unique():
            mask = self.df['genre'] == genre
            song_names = self._get_song_names(mask)
            
            fig.add_trace(
                go.Scatter(
                    x=X_umap2d[mask, 0],
                    y=X_umap2d[mask, 1],
                    mode='markers',
                    name=genre,
                    marker=dict(
                        size=8,
                        color=colors.get(genre, '#999999'),
                        line=dict(width=0.5, color='white')
                    ),
                    text=song_names,
                    hovertemplate='<b>%{text}</b><br>Genre: ' + genre + '<extra></extra>'
                )
            )
        
        fig.update_layout(
            title=f"Embedding Space: 2D UMAP Projection ({len(self.df)} songs)",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            width=1000,
            height=800,
            template='plotly_white',
            hovermode='closest'
        )
        
        if save:
            HTML_DIR.mkdir(parents=True, exist_ok=True)
            filepath = HTML_DIR / "embedding_space_2d.html"
            fig.write_html(filepath)
            print(f"✓ 2D visualization saved: {filepath}")
        
        fig.show()
        return fig
    
    def visualize_embedding_space_3d(self, save: bool = True):
        """
        3D UMAP projection with song titles on hover.
        """
        from sklearn.preprocessing import StandardScaler
        import umap
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.embeddings)
        
        umap_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap3d = umap_3d.fit_transform(X_scaled)
        
        colors = {'TRANCE': '#FF6B6B', 'ITALODANCE': '#4ECDC4'}
        
        fig = go.Figure()
        
        for genre in self.df['genre'].unique():
            mask = self.df['genre'] == genre
            song_names = self._get_song_names(mask)
            
            fig.add_trace(
                go.Scatter3d(
                    x=X_umap3d[mask, 0],
                    y=X_umap3d[mask, 1],
                    z=X_umap3d[mask, 2],
                    mode='markers',
                    name=genre,
                    marker=dict(
                        size=5,
                        color=colors.get(genre, '#999999'),
                        line=dict(width=0.5, color='white')
                    ),
                    text=song_names,
                    hovertemplate='<b>%{text}</b><br>Genre: ' + genre + '<extra></extra>'
                )
            )
        
        fig.update_layout(
            title=f"Embedding Space: 3D UMAP Projection ({len(self.df)} songs)",
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="UMAP 3"
            ),
            width=1200,
            height=900,
            template='plotly_white',
            hovermode='closest'
        )
        
        if save:
            HTML_DIR.mkdir(parents=True, exist_ok=True)
            filepath = HTML_DIR / "embedding_space_3d.html"
            fig.write_html(filepath)
            print(f"✓ 3D visualization saved: {filepath}")
        
        fig.show()
        return fig


def main():
    """Create all embedding-focused visualizations."""
    print("="*60)
    print("THE EMBEDDING JOURNEY: FROM MUSIC TO MATH")
    print("="*60)
    
    viz = EmbeddingJourneyVisualizer()
    
    # 1. Single song journey
    print("\n1. Creating single song transformation...")
    your_song_path = "/Users/danieleraimondi/MusicRecommenderSystem/data/raw/ITALODANCE/Nordic Stars - Crying in the rain (DanyR Italomelodic remix).mp3"
    
    if Path(your_song_path).exists():
        print(f"   Using: Nordic Stars - Crying in the rain")
        viz.visualize_single_song_journey(your_song_path, duration=None)
    else:
        print(f"   Warning: Song not found, using first available song")
        example_song = viz.df.iloc[0]
        viz.visualize_single_song_journey(example_song['file_path'], duration=None)
    
    # 2. 2D scatter
    print("\n2. Creating 2D UMAP scatter...")
    viz.visualize_embedding_space_2d()
    
    # 3. 3D scatter
    print("\n3. Creating 3D UMAP scatter...")
    viz.visualize_embedding_space_3d()
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)
    print(f"\nGenerated visualizations in {PLOTS_DIR}:")
    print("  1. embedding_journey_[song_name].png")
    print("  2. embedding_space_2d.html")
    print("  3. embedding_space_3d.html")
    print("\nEmbedding transformation complete 🎵")


if __name__ == "__main__":
    main()