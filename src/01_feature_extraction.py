"""
Audio feature extraction pipeline for music embedding generation.

This module extracts audio features from MP3 files using librosa,
producing fixed-length embedding vectors suitable for similarity search, clustering,
and music recommendation systems.
"""
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple, List

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    RAW_DATA_DIR, EMBEDDINGS_FILE, SONG_DATA_FILE,
    SAMPLE_RATE, N_MFCC
)


@dataclass
class AudioFeatures:
    """
    Container for extracted audio features.
    
    Attributes:
        features: Feature vector (numpy array)
        song_name: Song filename without extension
        file_path: Absolute path to audio file
        genre: Genre extracted from folder structure
        duration: Actual audio duration in seconds
    """
    features: np.ndarray
    song_name: str
    file_path: str
    genre: str
    duration: float


class AudioFeatureExtractor:
    """
    Audio feature extractor for music analysis.
    
    Extracts multi-dimensional audio features including timbral, harmonic,
    spectral, rhythmic, and energy characteristics from audio files.
    Processes entire songs without duration limits.
    
    Attributes:
        sr: Target sample rate (Hz)
        n_mfcc: Number of MFCC coefficients to extract
        feature_dim: Dimensionality of output feature vectors
    """
    
    def __init__(self, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC):
        """
        Initialize feature extractor.
        
        Args:
            sr: Sample rate for audio processing (default from config)
            n_mfcc: Number of MFCC coefficients (default from config)
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.feature_dim = None
    
    def extract_features(self, file_path: Path) -> Optional[AudioFeatures]:
        """
        Extract feature vector from audio file.
        
        Processes the entire audio file and extracts:
        - MFCC (Mel-Frequency Cepstral Coefficients): timbre and texture
        - Chroma: harmonic and tonal content
        - Spectral features: brightness, energy distribution
        - Rhythm: tempo and zero-crossing rate
        - Energy: RMS and dynamics
        - Tonnetz: harmonic relationships
        - Mel spectrogram: perceptual frequency representation
        
        Args:
            file_path: Path object pointing to audio file
            
        Returns:
            AudioFeatures object containing feature vector and metadata,
            or None if extraction fails
        """
        try:
            # Load entire audio file
            y, sr = librosa.load(file_path, sr=self.sr, duration=None)
            
            if len(y) == 0:
                print(f"Empty audio: {file_path.name}")
                return None
            
            actual_duration = len(y) / sr
            
            # Skip files shorter than 3 seconds (insufficient for robust features)
            if actual_duration < 3.0:
                print(f"Too short ({actual_duration:.1f}s): {file_path.name}")
                return None
            
            # 1. Timbral features (MFCC)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_delta = librosa.feature.delta(mfcc)  # Temporal variations
            
            # 2. Harmonic features (Chroma)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # 3. Spectral features
            spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spec_flatness = librosa.feature.spectral_flatness(y=y)
            
            # 4. Rhythmic features
            zcr = librosa.feature.zero_crossing_rate(y)
            
            # Robust tempo extraction with fallback
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                if np.isnan(tempo) or tempo == 0:
                    tempo = 120.0  # Fallback default
            except Exception:
                tempo = 120.0
            
            # 5. Energy features
            rms = librosa.feature.rms(y=y)
            
            # 6. Harmonic network (Tonnetz)
            try:
                harmonic = librosa.effects.harmonic(y)
                if len(harmonic) > 0:
                    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
                else:
                    tonnetz = np.zeros((6, 1))  # Fallback
            except Exception:
                tonnetz = np.zeros((6, 1))
            
            # 7. Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Statistical aggregation: reduce temporal dimension to fixed vector
            # Build feature list with explicit shape control
            feature_list = []
            
            # Helper function to ensure proper array format
            def safe_append(value, is_scalar=False):
                """Safely convert value to 1D array"""
                if is_scalar:
                    return np.array([value], dtype=np.float64)
                else:
                    arr = np.asarray(value, dtype=np.float64)
                    return arr.flatten() if arr.ndim > 1 else arr
            
            # MFCC statistics (60 features: 20 mean + 20 std + 20 delta)
            feature_list.append(safe_append(np.mean(mfcc, axis=1)))
            feature_list.append(safe_append(np.std(mfcc, axis=1)))
            feature_list.append(safe_append(np.mean(mfcc_delta, axis=1)))
            
            # Chroma statistics (24 features: 12 mean + 12 std)
            feature_list.append(safe_append(np.mean(chroma, axis=1)))
            feature_list.append(safe_append(np.std(chroma, axis=1)))
            
            # Spectral statistics (16 features)
            feature_list.append(safe_append(np.mean(spec_centroid), is_scalar=True))
            feature_list.append(safe_append(np.std(spec_centroid), is_scalar=True))
            feature_list.append(safe_append(np.mean(spec_rolloff), is_scalar=True))
            feature_list.append(safe_append(np.std(spec_rolloff), is_scalar=True))
            feature_list.append(safe_append(np.mean(spec_bandwidth), is_scalar=True))
            feature_list.append(safe_append(np.std(spec_bandwidth), is_scalar=True))
            feature_list.append(safe_append(np.mean(spec_contrast, axis=1)))
            feature_list.append(safe_append(np.std(spec_contrast, axis=1)))
            feature_list.append(safe_append(np.mean(spec_flatness), is_scalar=True))
            
            # Rhythmic statistics (3 features)
            feature_list.append(safe_append(np.mean(zcr), is_scalar=True))
            feature_list.append(safe_append(np.std(zcr), is_scalar=True))
            feature_list.append(safe_append(tempo, is_scalar=True))
            
            # Energy statistics (2 features)
            feature_list.append(safe_append(np.mean(rms), is_scalar=True))
            feature_list.append(safe_append(np.std(rms), is_scalar=True))
            
            # Tonnetz statistics (12 features: 6 mean + 6 std)
            feature_list.append(safe_append(np.mean(tonnetz, axis=1)))
            feature_list.append(safe_append(np.std(tonnetz, axis=1)))
            
            # Mel spectrogram statistics (512 features: 128 mean + 384 percentiles)
            feature_list.append(safe_append(np.mean(mel_spec_db, axis=1)))
            feature_list.append(safe_append(np.percentile(mel_spec_db, 25, axis=1)))
            feature_list.append(safe_append(np.percentile(mel_spec_db, 50, axis=1)))
            feature_list.append(safe_append(np.percentile(mel_spec_db, 75, axis=1)))
            
            # Concatenate all features
            features = np.concatenate(feature_list)
            
            return AudioFeatures(
                features=features,
                song_name=file_path.stem,
                file_path=str(file_path),
                genre=self._extract_genre(file_path),
                duration=actual_duration
            )
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            return None
    
    def _extract_genre(self, file_path: Path) -> str:
        """
        Extract genre label from folder structure.
        
        Assumes folder structure: RAW_DATA_DIR/GENRE/song.mp3
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Genre string (uppercase) or "UNKNOWN"
        """
        try:
            relative = file_path.relative_to(RAW_DATA_DIR)
            return relative.parts[0].upper() if len(relative.parts) > 1 else "UNKNOWN"
        except (ValueError, IndexError):
            return "UNKNOWN"
    
    def process_parallel(
        self, 
        mp3_files: List[Path], 
        max_workers: int = 4
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Process multiple audio files in parallel.
        
        Uses thread-based parallelism for I/O-bound audio loading operations.
        Automatically handles inconsistent feature dimensions.
        
        Args:
            mp3_files: List of Path objects to audio files
            max_workers: Number of parallel workers (default: 4)
            
        Returns:
            Tuple of (metadata_dataframe, embeddings_matrix)
            
        Raises:
            ValueError: If no features were successfully extracted
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.extract_features, fp): fp for fp in mp3_files}
            
            for future in tqdm(
                as_completed(futures), 
                total=len(mp3_files), 
                desc="Extracting features"
            ):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        if not results:
            raise ValueError("No features successfully extracted")
        
        # Validate feature dimensionality consistency
        feature_dims = [len(r.features) for r in results]
        if len(set(feature_dims)) > 1:
            print(f"WARNING: Inconsistent feature dimensions detected: {set(feature_dims)}")
            target_dim = max(set(feature_dims), key=feature_dims.count)
            results = [r for r in results if len(r.features) == target_dim]
            print(f"Filtered to {len(results)} songs with dimension {target_dim}")
        
        self.feature_dim = len(results[0].features)
        
        # Build metadata dataframe
        df = pd.DataFrame({
            'song_name': [r.song_name for r in results],
            'file_path': [r.file_path for r in results],
            'genre': [r.genre for r in results],
            'duration': [r.duration for r in results]
        })
        
        # Build embedding matrix
        embeddings = np.vstack([r.features for r in results])
        
        return df, embeddings


def load_mp3_files(folder_path: Path) -> List[Path]:
    """
    Recursively load all MP3 files from folder.
    
    Args:
        folder_path: Root directory to search
        
    Returns:
        List of Path objects to MP3 files
    """
    mp3_files = list(Path(folder_path).glob("**/*.mp3"))
    print(f"Found {len(mp3_files)} MP3 files")
    return mp3_files


def save_embeddings(df: pd.DataFrame, embeddings: np.ndarray) -> None:
    """
    Save embeddings and metadata to disk with validation.
    
    Args:
        df: Metadata dataframe
        embeddings: Feature matrix (n_songs x n_features)
        
    Raises:
        AssertionError: If data validation fails
    """
    # Validation checks
    assert len(df) == len(embeddings), \
        f"Mismatch: {len(df)} metadata rows vs {len(embeddings)} embedding rows"
    assert not df['song_name'].duplicated().any(), \
        "Duplicate song names detected"
    
    # Create directories if needed
    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SONG_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to disk
    df.to_pickle(SONG_DATA_FILE)
    np.save(EMBEDDINGS_FILE, embeddings)
    
    print(f"\n✓ Embeddings saved: {EMBEDDINGS_FILE}")
    print(f"✓ Metadata saved: {SONG_DATA_FILE}")


def print_statistics(df: pd.DataFrame, embeddings: np.ndarray) -> None:
    """
    Print descriptive statistics of extracted features.
    
    Args:
        df: Metadata dataframe
        embeddings: Feature matrix
    """
    print(f"\n{'='*60}")
    print(f"EXTRACTION STATISTICS")
    print(f"{'='*60}")
    print(f"Total songs processed: {len(df)}")
    print(f"Feature space dimension: {embeddings.shape[1]}")
    print(f"Average duration: {df['duration'].mean():.1f}s (±{df['duration'].std():.1f}s)")
    print(f"Total audio time: {df['duration'].sum()/60:.1f} minutes")
    
    print(f"\nGenre distribution:")
    for genre, count in df['genre'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {genre:15s} {count:4d} ({pct:5.1f}%)")
    
    print(f"\nEmbedding statistics:")
    print(f"  Mean:  {embeddings.mean():8.4f}")
    print(f"  Std:   {embeddings.std():8.4f}")
    print(f"  Min:   {embeddings.min():8.4f}")
    print(f"  Max:   {embeddings.max():8.4f}")
    print(f"  Range: {embeddings.max() - embeddings.min():8.4f}")


def main():
    """
    Main execution pipeline for audio feature extraction.
    
    Pipeline steps:
    1. Load MP3 files from configured directory
    2. Extract features in parallel
    3. Validate and aggregate results
    4. Save embeddings and metadata
    5. Print statistics
    """
    print("="*60)
    print("AUDIO FEATURE EXTRACTION PIPELINE")
    print("="*60)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"MFCC coefficients: {N_MFCC}")
    print(f"Processing: FULL SONG (no duration limit)")
    print(f"Source directory: {RAW_DATA_DIR}")
    
    # Load audio files
    mp3_files = load_mp3_files(RAW_DATA_DIR)
    
    if not mp3_files:
        print(f"\nERROR: No MP3 files found in {RAW_DATA_DIR}")
        print("Place your MP3 files in data/raw/ folder")
        return
    
    # Extract features
    extractor = AudioFeatureExtractor()
    df, embeddings = extractor.process_parallel(mp3_files, max_workers=4)
    
    # Display statistics
    print_statistics(df, embeddings)
    
    # Save results
    save_embeddings(df, embeddings)
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()