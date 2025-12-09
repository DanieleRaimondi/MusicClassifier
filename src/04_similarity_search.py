"""
Music Similarity Search Engine.

Finds similar songs based on audio embeddings using cosine similarity.
Optimized for speed and consistency with classification pipeline.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import sys
import joblib
from typing import Optional, List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import EMBEDDINGS_FILE, SONG_DATA_FILE, PLOTS_DIR
except ModuleNotFoundError:
    BASE_DIR = Path(__file__).parent.parent
    EMBEDDINGS_FILE = BASE_DIR / "data" / "processed" / "embeddings_matrix.npy"
    SONG_DATA_FILE = BASE_DIR / "data" / "processed" / "song_data.pkl"
    PLOTS_DIR = BASE_DIR / "data" / "plots"


class MusicSimilaritySearch:
    """
    Similarity search engine for music recommendations.
    
    Features:
    - Precomputed similarity matrix for speed
    - Consistent preprocessing with classifier
    - Diverse recommendation algorithm (MMR)
    - File path support for playback integration
    """
    
    def __init__(self, use_classifier_preprocessing: bool = False, precompute_similarities: bool = True):
        """
        Initialize similarity search engine.
        
        Args:
            use_classifier_preprocessing: Use same scaler+PCA from trained classifier
            precompute_similarities: Precompute full similarity matrix (faster queries)
        """
        self.df = pd.read_pickle(SONG_DATA_FILE)
        self.embeddings = np.load(EMBEDDINGS_FILE)
        
        self.song_to_idx = {name: idx for idx, name in enumerate(self.df['song_name'])}
        
        # Preprocessing
        if use_classifier_preprocessing:
            model_path = PLOTS_DIR.parent / "models" / "best_genre_classifier.pkl"
            if model_path.exists():
                print("Loading classifier preprocessing pipeline...")
                pipeline = joblib.load(model_path)
                scaler = pipeline['scaler']
                pca = pipeline.get('pca', None)
                
                embeddings_scaled = scaler.transform(self.embeddings)
                if pca is not None:
                    self.embeddings_normalized = pca.transform(embeddings_scaled)
                    print(f"   Using PCA: {self.embeddings_normalized.shape[1]}D")
                else:
                    self.embeddings_normalized = embeddings_scaled
            else:
                print(f"⚠️  Classifier not found at {model_path}, using StandardScaler")
                scaler = StandardScaler()
                self.embeddings_normalized = scaler.fit_transform(self.embeddings)
        else:
            scaler = StandardScaler()
            self.embeddings_normalized = scaler.fit_transform(self.embeddings)
        
        # Precompute similarities for fast queries
        self.similarity_matrix = None
        if precompute_similarities:
            print("Precomputing similarity matrix...")
            self.similarity_matrix = cosine_similarity(self.embeddings_normalized)
            print(f"   Matrix shape: {self.similarity_matrix.shape}")
        
        print(f"✓ Loaded {len(self.df)} songs with {self.embeddings_normalized.shape[1]}D embeddings\n")
    
    def find_similar(
        self,
        query: str,
        n: int = 10,
        same_genre: bool = False,
        exclude_query: bool = True
    ) -> pd.DataFrame:
        """
        Find n most similar songs to query.
        
        Args:
            query: Song name (exact or substring)
            n: Number of results
            same_genre: Restrict to same genre
            exclude_query: Exclude query song from results
            
        Returns:
            DataFrame with ranked similar songs
        """
        query_idx = self._find_song_index(query)
        if query_idx is None:
            raise ValueError(f"Song not found: {query}")
        
        query_song = self.df.iloc[query_idx]['song_name']
        query_genre = self.df.iloc[query_idx]['genre']
        
        print(f"Query: {query_song}")
        print(f"Genre: {query_genre}\n")
        
        # Get similarities
        if self.similarity_matrix is not None:
            similarities = self.similarity_matrix[query_idx].copy()
        else:
            query_emb = self.embeddings_normalized[query_idx].reshape(1, -1)
            similarities = cosine_similarity(query_emb, self.embeddings_normalized)[0]
        
        # Filter by genre if needed
        valid_mask = np.ones(len(self.df), dtype=bool)
        if same_genre:
            valid_mask &= (self.df['genre'] == query_genre).values
        
        # Exclude query
        if exclude_query:
            valid_mask[query_idx] = False
        
        valid_idx = np.where(valid_mask)[0]
        valid_similarities = similarities[valid_idx]
        
        # Get top n
        top_n_local = np.argsort(valid_similarities)[::-1][:n]
        top_idx = valid_idx[top_n_local]
        top_sim = valid_similarities[top_n_local]
        
        # Build results
        results = pd.DataFrame({
            'rank': range(1, len(top_idx) + 1),
            'song_name': self.df.iloc[top_idx]['song_name'].values,
            'genre': self.df.iloc[top_idx]['genre'].values,
            'similarity': top_sim,
            'file_path': self.df.iloc[top_idx]['file_path'].values
        })
        
        if 'duration' in self.df.columns:
            results['duration'] = self.df.iloc[top_idx]['duration'].values
        
        return results
    
    def find_diverse(
        self, 
        query: str, 
        n: int = 10, 
        diversity: float = 0.3,
        method: str = 'mmr'
    ) -> pd.DataFrame:
        """
        Find recommendations balancing similarity and diversity.
        
        Uses Maximal Marginal Relevance (MMR) algorithm for
        diverse recommendation.
        
        Args:
            query: Song name
            n: Number of results
            diversity: Diversity weight [0=pure similarity, 1=pure diversity]
            method: 'mmr' (fast) or 'greedy' (accurate but slow)
            
        Returns:
            DataFrame with diverse recommendations
        """
        query_idx = self._find_song_index(query)
        if query_idx is None:
            raise ValueError(f"Song not found: {query}")
        
        # Get similarities to query
        if self.similarity_matrix is not None:
            sim_to_query = self.similarity_matrix[query_idx].copy()
        else:
            query_emb = self.embeddings_normalized[query_idx].reshape(1, -1)
            sim_to_query = cosine_similarity(query_emb, self.embeddings_normalized)[0]
        
        # Exclude query
        candidate_mask = np.ones(len(self.df), dtype=bool)
        candidate_mask[query_idx] = False
        
        if method == 'mmr':
            selected = self._mmr_selection(sim_to_query, candidate_mask, n, diversity)
        else:
            selected = self._greedy_selection(sim_to_query, candidate_mask, n, diversity)
        
        # Build results
        results = pd.DataFrame({
            'rank': range(1, len(selected) + 1),
            'song_name': self.df.iloc[selected]['song_name'].values,
            'genre': self.df.iloc[selected]['genre'].values,
            'similarity': sim_to_query[selected],
            'file_path': self.df.iloc[selected]['file_path'].values
        })
        
        if 'duration' in self.df.columns:
            results['duration'] = self.df.iloc[selected]['duration'].values
        
        return results
    
    def _mmr_selection(
        self, 
        sim_to_query: np.ndarray, 
        candidate_mask: np.ndarray, 
        n: int, 
        diversity: float
    ) -> List[int]:
        """
        Maximal Marginal Relevance selection.
        
        Vectorized implementation: O(n * m) instead of O(n * m^2)
        """
        selected = []
        remaining_mask = candidate_mask.copy()
        
        for _ in range(n):
            remaining_idx = np.where(remaining_mask)[0]
            if len(remaining_idx) == 0:
                break
            
            # Similarity to query
            relevance = sim_to_query[remaining_idx]
            
            if len(selected) == 0:
                # First selection: pure relevance
                best_local = np.argmax(relevance)
            else:
                # Similarity to already selected (vectorized)
                if self.similarity_matrix is not None:
                    sim_to_selected = self.similarity_matrix[remaining_idx][:, selected]
                else:
                    remaining_emb = self.embeddings_normalized[remaining_idx]
                    selected_emb = self.embeddings_normalized[selected]
                    sim_to_selected = cosine_similarity(remaining_emb, selected_emb)
                
                # MMR score: balance relevance and diversity
                max_sim_to_selected = sim_to_selected.max(axis=1)
                mmr_scores = (1 - diversity) * relevance - diversity * max_sim_to_selected
                best_local = np.argmax(mmr_scores)
            
            best_idx = remaining_idx[best_local]
            selected.append(best_idx)
            remaining_mask[best_idx] = False
        
        return selected
    
    def _greedy_selection(
        self, 
        sim_to_query: np.ndarray, 
        candidate_mask: np.ndarray, 
        n: int, 
        diversity: float
    ) -> List[int]:
        """Greedy selection (slower but more accurate for small n)."""
        selected = []
        remaining_idx = list(np.where(candidate_mask)[0])
        
        for _ in range(n):
            if not remaining_idx:
                break
            
            scores = []
            for idx in remaining_idx:
                relevance = sim_to_query[idx]
                
                if selected:
                    if self.similarity_matrix is not None:
                        sim_to_selected = self.similarity_matrix[idx, selected]
                    else:
                        candidate_emb = self.embeddings_normalized[idx].reshape(1, -1)
                        selected_emb = self.embeddings_normalized[selected]
                        sim_to_selected = cosine_similarity(candidate_emb, selected_emb)[0]
                    
                    max_sim = sim_to_selected.max()
                    score = (1 - diversity) * relevance - diversity * max_sim
                else:
                    score = relevance
                
                scores.append(score)
            
            best_local = np.argmax(scores)
            best_idx = remaining_idx[best_local]
            selected.append(best_idx)
            remaining_idx.remove(best_idx)
        
        return selected
    
    def compare_songs(self, song1: str, song2: str) -> Dict:
        """
        Compare two songs.
        
        Args:
            song1: First song name
            song2: Second song name
            
        Returns:
            Dictionary with comparison metrics
        """
        idx1 = self._find_song_index(song1)
        idx2 = self._find_song_index(song2)
        
        if idx1 is None or idx2 is None:
            raise ValueError("One or both songs not found")
        
        if self.similarity_matrix is not None:
            cosine_sim = self.similarity_matrix[idx1, idx2]
        else:
            emb1 = self.embeddings_normalized[idx1].reshape(1, -1)
            emb2 = self.embeddings_normalized[idx2].reshape(1, -1)
            cosine_sim = cosine_similarity(emb1, emb2)[0, 0]
        
        return {
            'song1': self.df.iloc[idx1]['song_name'],
            'song2': self.df.iloc[idx2]['song_name'],
            'genre1': self.df.iloc[idx1]['genre'],
            'genre2': self.df.iloc[idx2]['genre'],
            'cosine_similarity': cosine_sim,
            'file_path1': self.df.iloc[idx1]['file_path'],
            'file_path2': self.df.iloc[idx2]['file_path']
        }
    
    def get_genre_statistics(self) -> pd.DataFrame:
        """Get statistics about genre similarity."""
        genres = self.df['genre'].unique()
        stats = []
        
        for genre in genres:
            mask = self.df['genre'] == genre
            genre_idx = np.where(mask)[0]
            
            if self.similarity_matrix is not None:
                # Intra-genre similarity
                intra_sim = self.similarity_matrix[np.ix_(genre_idx, genre_idx)]
                np.fill_diagonal(intra_sim, np.nan)
                intra_mean = np.nanmean(intra_sim)
                
                # Inter-genre similarity
                other_mask = ~mask
                other_idx = np.where(other_mask)[0]
                if len(other_idx) > 0:
                    inter_sim = self.similarity_matrix[np.ix_(genre_idx, other_idx)]
                    inter_mean = inter_sim.mean()
                else:
                    inter_mean = np.nan
            else:
                intra_mean = np.nan
                inter_mean = np.nan
            
            stats.append({
                'genre': genre,
                'n_songs': len(genre_idx),
                'intra_similarity': intra_mean,
                'inter_similarity': inter_mean
            })
        
        return pd.DataFrame(stats)
    
    def list_by_genre(self, genre: str, n: int = 20) -> pd.DataFrame:
        """Get songs from specific genre."""
        mask = self.df['genre'].str.upper() == genre.upper()
        df_genre = self.df[mask]
        
        if len(df_genre) == 0:
            raise ValueError(f"No songs found for genre: {genre}")
        
        return df_genre.sample(n=min(n, len(df_genre))).reset_index(drop=True)
    
    def _find_song_index(self, query: str) -> Optional[int]:
        """Find song index by exact match or substring."""
        # Exact match
        if query in self.song_to_idx:
            return self.song_to_idx[query]
        
        # Substring search (case-insensitive)
        matches = self.df[self.df['song_name'].str.contains(query, case=False, na=False)]
        
        if len(matches) == 0:
            return None
        elif len(matches) == 1:
            return matches.index[0]
        else:
            print(f"\n⚠️  Multiple matches for '{query}':")
            for idx, row in matches.head(10).iterrows():
                print(f"  {idx}: {row['song_name']}")
            print("\nUse exact name or more specific query")
            return None
    
    def list_all(self, genre: Optional[str] = None) -> pd.DataFrame:
        """List all songs, optionally filtered by genre."""
        if genre:
            mask = self.df['genre'].str.upper() == genre.upper()
            df_filtered = self.df[mask]
        else:
            df_filtered = self.df
        
        cols = ['song_name', 'genre']
        if 'duration' in df_filtered.columns:
            cols.append('duration')
        cols.append('file_path')
        
        return df_filtered[cols].sort_values('song_name').reset_index(drop=True)


def interactive():
    """Interactive CLI for similarity search."""
    print("="*60)
    print("MUSIC SIMILARITY SEARCH")
    print("="*60)
    
    engine = MusicSimilaritySearch(
        use_classifier_preprocessing=True,
        precompute_similarities=True
    )
    
    print("Commands:")
    print("  search <song>               Find similar songs")
    print("  diverse <song>              Find diverse recommendations")
    print("  compare <song1> | <song2>   Compare two songs")
    print("  list [genre]                List songs")
    print("  genres                      Show genre statistics")
    print("  quit                        Exit\n")
    
    while True:
        try:
            cmd = input(">>> ").strip()
            
            if cmd.lower() in ['quit', 'exit', 'q']:
                break
            
            elif cmd.lower().startswith('search '):
                query = cmd[7:].strip()
                if query:
                    results = engine.find_similar(query, n=10)
                    print(results[['rank', 'song_name', 'genre', 'similarity']].to_string(index=False))
                    print()
            
            elif cmd.lower().startswith('diverse '):
                query = cmd[8:].strip()
                if query:
                    results = engine.find_diverse(query, n=10, diversity=0.3)
                    print(results[['rank', 'song_name', 'genre', 'similarity']].to_string(index=False))
                    print()
            
            elif cmd.lower().startswith('compare '):
                songs = cmd[8:].split('|')
                if len(songs) == 2:
                    comp = engine.compare_songs(songs[0].strip(), songs[1].strip())
                    print(f"\n{comp['song1']} ({comp['genre1']})")
                    print(f"  vs")
                    print(f"{comp['song2']} ({comp['genre2']})")
                    print(f"\nCosine similarity: {comp['cosine_similarity']:.4f}\n")
                else:
                    print("Usage: compare <song1> | <song2>\n")
            
            elif cmd.lower().startswith('list'):
                parts = cmd.split(maxsplit=1)
                genre = parts[1].strip() if len(parts) > 1 else None
                songs = engine.list_all(genre)
                print(f"\n{len(songs)} songs:")
                print(songs[['song_name', 'genre']].head(20).to_string(index=False))
                if len(songs) > 20:
                    print(f"... and {len(songs) - 20} more\n")
                else:
                    print()
            
            elif cmd.lower() == 'genres':
                stats = engine.get_genre_statistics()
                print("\nGenre statistics:")
                print(stats.to_string(index=False))
                print()
            
            else:
                print("Unknown command\n")
        
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Demo examples."""
    print("="*60)
    print("SIMILARITY SEARCH DEMO")
    print("="*60)
    
    engine = MusicSimilaritySearch(
        use_classifier_preprocessing=True,
        precompute_similarities=True
    )
    
    # Example 1: Find similar
    print("\nExample 1: Find similar songs")
    print("-"*60)
    example = engine.df.iloc[0]['song_name']
    results = engine.find_similar(example, n=10)
    print(results[['rank', 'song_name', 'genre', 'similarity']].to_string(index=False))
    
    # Example 2: Compare
    if len(engine.df) >= 2:
        print("\n\nExample 2: Compare songs")
        print("-"*60)
        song1 = engine.df.iloc[0]['song_name']
        song2 = engine.df.iloc[10]['song_name']
        comp = engine.compare_songs(song1, song2)
        print(f"{comp['song1']} ({comp['genre1']})")
        print(f"  vs")
        print(f"{comp['song2']} ({comp['genre2']})")
        print(f"Cosine: {comp['cosine_similarity']:.4f}")
    
    # Example 3: Diverse
    print("\n\nExample 3: Diverse recommendations (MMR)")
    print("-"*60)
    diverse = engine.find_diverse(example, n=10, diversity=0.3, method='mmr')
    print(diverse[['rank', 'song_name', 'genre', 'similarity']].to_string(index=False))
    
    # Example 4: Genre statistics
    print("\n\nExample 4: Genre statistics")
    print("-"*60)
    stats = engine.get_genre_statistics()
    print(stats.to_string(index=False))
    
    print("\n" + "="*60)
    print("For interactive mode: python -c 'from similarity_search import interactive; interactive()'")
    print("="*60)


if __name__ == "__main__":
    interactive()