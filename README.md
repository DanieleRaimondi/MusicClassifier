# 🎵 Music Embedding Analysis: From Producer to Data Scientist

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![librosa](https://img.shields.io/badge/librosa-0.10.0-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-red.svg)

*Bridging two worlds: music production and machine learning*

</div>

---

## 📖 Project Overview

This project represents a personal journey spanning over a decade, from producing Italodance remixes in Fruity Loops to analyzing music through the lens of data science. It explores how machines "understand" music by transforming audio into numerical representations, enabling the same sophisticated recommendation and analysis capabilities used by popular platforms.

### 🎼 The Story

In the early 2010s, I was passionate about electronic music production, particularly **Italodance** and **Trance** genres. I spent countless hours in Fruity Loops Studio crafting remixes, including my "Nordic Stars - Crying in the Rain (DanyR Italomelodic Remix)" in 2014.

Fast forward to today: as a **Data Scientist**, I became curious about how computers analyze and understand music. This small project combines both passions, music and data science, to create a bridge between artistic intuition and machine learning.

**The dataset**: ~300 copyright-free songs from my personal collection (Italodance and Trance), including my own 2014 remix.

**The goal**: Transform audio signals into numerical embeddings, build a genre classifier with explainability, create a similarity search engine, and visualize how machines perceive musical relationships—ultimately understanding the technology behind modern music recommendation systems.

---

## 🎯 What This Project Does

### 1. **Audio Feature Extraction** (`01_feature_extraction.py`)
- Loads entire audio files using `librosa`
- Extracts **641-dimensional hand-crafted audio embeddings (not derived from neural networks)** capturing:
  - **Timbral features**: 20 MFCCs + deltas (60D) - sound texture and instrument characteristics
  - **Harmonic features**: Chroma CQT (24D) - pitch class distribution and chord structure
  - **Spectral features**: Centroid, rolloff, bandwidth, contrast, flatness (16D) - brightness and energy distribution
  - **Rhythmic features**: Zero-crossing rate, tempo (3D) - beat structure and rhythm patterns
  - **Energy features**: RMS mean/std (2D) - dynamics and loudness
  - **Harmonic network**: Tonnetz (12D) - tonal relationships
  - **Mel spectrogram statistics**: 128 mel bins × 4 aggregations (512D) - perceptual frequency representation
- Parallel processing with ThreadPoolExecutor (4 workers)
- Error handling and feature dimension validation
- Saves embeddings matrix (`.npy`) and metadata (`.pkl`)

### 2. **Embedding Journey Visualization** (`02_embedding_journey.py`)
- **Single song transformation**: Shows the complete pipeline from waveform → spectrogram → features → 641D embedding
- **2D UMAP projection**: Interactive scatter plot
- **3D UMAP projection**: Rotatable 3D visualization of embedding space
- Matches production pipeline exactly (641D consistency)
- High-quality PNG exports + interactive HTML plots

### 3. **Genre Classification with SHAP** (`03_genre_classifier.py`)
- **XGBoost classifier** with aggressive regularization to prevent overfitting
- **PCA dimensionality reduction** (641D → 15D) retaining ~85% variance
- **Train/test split** with no data leakage (scaler fit only on train)
- **5-fold cross-validation** to monitor generalization
- **SHAP (SHapley Additive exPlanations)** for model interpretability:
  - Global feature importance (bar + beeswarm plots)
  - Individual song explanations with waterfall plots
  - Natural language explanations mapping PCs to musical concepts
  - Dependence plots showing feature interactions
- **Boundary song analysis**: Identifies songs with mixed characteristics
- Model serialization for eventual deployment

### 4. **Similarity Search Engine** (`04_similarity_search.py`)
- **Cosine similarity** in normalized embedding space
- **Precomputed similarity matrix** for instant queries
- **Standard search**: Top-N most similar songs
- **Diverse recommendations**: MMR (Maximal Marginal Relevance) algorithm balancing similarity and diversity
- **Song comparison**: Direct similarity measurement between any two tracks
- **Genre statistics**: Intra-genre vs inter-genre similarity analysis
- **Interactive CLI** for exploration
- File path support for playback integration

---

## 🚀 Real-World Applications

Once music is transformed into numerical embeddings, numerous applications become possible:

### 🎵 **Music Recommendation Systems**
- **Content-based filtering**: Find songs similar to a given track by computing cosine similarity in embedding space
- **Collaborative filtering**: Combine user listening patterns with audio features for personalized recommendations
- **Cold-start solution**: Recommend new/unpopular songs based solely on audio characteristics (no listening history needed)
- **Hybrid systems**: Merge audio similarity with user behavior for optimal recommendations

### 🔍 **Similarity Search & Discovery**
- Build nearest-neighbor search indices (FAISS, Annoy) for instant "songs like this" queries at scale
- Create **dynamic playlists** that smoothly transition between similar tracks
- **Discover hidden connections** between artists and genres through embedding proximity
- **Radio mode**: Generate infinite playlists starting from a seed song

### 🏷️ **Automatic Genre Classification**
- Train supervised models (Random Forest, XGBoost, Neural Networks) to predict genres
- **Multi-label classification** for songs spanning multiple genres
- **Sub-genre detection** and fine-grained style taxonomy creation
- **Mood classification**: Predict energy, valence, danceability, emotional tone

### 📊 **Music Analysis & Insights**
- **Trend analysis**: Track how musical styles evolve over time (spectral features, tempo changes)
- **Audio fingerprinting**: Identify songs from short audio clips (e.g., Shazam-like functionality)
- **Cover song detection**: Find different versions of the same composition using harmonic features
- **Plagiarism detection**: Identify melodic or harmonic similarities between tracks
- **A&R analytics**: Benchmark new productions against successful reference tracks

### 🎛️ **Production & Creative Tools**
- Analyze successful tracks to identify common patterns (e.g., "what makes a hit?")
- **Reference track matching**: Find commercial songs with similar characteristics to guide mixing/mastering
- Predict commercial potential based on audio features + historical data
- Assist A&R teams in discovering emerging talent through clustering analysis

### 🤖 **Generative AI Applications**
- Generate **latent space interpolations** between tracks (smooth transitions)
- **Style transfer**: Apply the "sonic fingerprint" of one song to another
- Train **music generation models** (VAE, GAN, diffusion models) using embeddings as conditioning
- **Playlist generation**: AI-curated playlists optimized for specific moods or activities

### 📱 **User Experience Features**
- **Smart shuffle**: Playlist ordering that maximizes musical coherence
- **Transition analysis**: Smooth DJ-style crossfades between compatible tracks
- **Workout/study playlist optimization**: Select songs matching target BPM and energy levels
- **Discover Weekly**: Automated weekly playlists based on listening history + audio similarity

---

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Audio Processing** | librosa 0.10.0 | Loading, resampling, feature extraction |
| **Feature Extraction** | librosa | MFCCs, chroma, spectral, rhythmic, harmonic features |
| **ML Framework** | scikit-learn 1.3.0 | Preprocessing, PCA, cross-validation |
| **Classification** | XGBoost 2.0.0 | Genre classification with gradient boosting |
| **Explainability** | SHAP 0.43.0 | Model interpretation and feature importance |
| **Dimensionality Reduction** | UMAP 0.5.4 | Nonlinear projection for visualization |
| **Similarity Search** | scikit-learn | Cosine similarity, MMR algorithm |
| **Visualization** | matplotlib, seaborn, plotly | Static plots + interactive HTML |
| **Data Handling** | pandas, numpy | Dataframes, matrices, serialization |

---

## 🔬 Feature Engineering Details

The **641-dimensional embedding** captures comprehensive audio characteristics:

### 1. **Timbral Features** (60D)
- **20 MFCCs** (mean): Mel-Frequency Cepstral Coefficients - sound texture
- **20 MFCCs** (std): Temporal variation in timbre
- **20 MFCC deltas** (mean): Rate of timbral change

### 2. **Harmonic Features** (24D)
- **12 Chroma** (mean): Pitch class distribution across 12 semitones
- **12 Chroma** (std): Harmonic variation over time

### 3. **Spectral Features** (16D)
- **Spectral centroid** (mean, std): Brightness / "center of mass" of spectrum
- **Spectral rolloff** (mean, std): Frequency below which 85% of energy is contained
- **Spectral bandwidth** (mean, std): Width of frequency distribution
- **7 Spectral contrast bands** (mean): Energy difference between peaks and valleys
- **7 Spectral contrast bands** (std): Variation in spectral contrast
- **Spectral flatness** (mean): Noisiness vs tonality

### 4. **Rhythmic Features** (3D)
- **Zero-crossing rate** (mean, std): Number of sign changes in signal
- **Tempo** (BPM): Estimated beats per minute

### 5. **Energy Features** (2D)
- **RMS energy** (mean, std): Root mean square amplitude - perceived loudness

### 6. **Harmonic Network** (12D)
- **6 Tonnetz coordinates** (mean): Harmonic relationships in tonal space
- **6 Tonnetz coordinates** (std): Variation in harmonic structure

### 7. **Mel Spectrogram Statistics** (512D)
- **128 mel bins** (mean): Average energy per frequency band
- **128 mel bins** (25th percentile): Lower energy distribution
- **128 mel bins** (50th percentile - median): Central energy tendency
- **128 mel bins** (75th percentile): Upper energy distribution

### Statistical Aggregation Strategy
All temporal features are aggregated using **mean** and **std** (or percentiles for mel spectrogram) to produce fixed-length vectors regardless of song duration. This ensures:
- Consistent dimensionality across all songs
- Capture of both average characteristics and variation
- Robustness to different song lengths

---

## 🎓 What I Learned

### Technical Insights
- **Audio ≠ what we hear**: Raw waveforms contain far more information than human perception processes consciously
- **High-dimensional embeddings work**: 641D captures nuanced differences, but PCA to 15D retains 85% of variance with better generalization
- **Regularization is critical**: Electronic music genres overlap significantly—aggressive regularization prevents memorization
- **SHAP reveals musical intuition**: Features the model uses (e.g., PC3 for melodic directness) align with what producers adjust in practice

### Conceptual Bridges
- **Producer intuition ≈ Feature importance**: What producers tweak (EQ, reverb, compression) directly maps to spectral/temporal features
- **Genre boundaries are fuzzy**: Embedding space reveals gradual transitions rather than hard category separations—many songs are "hybrids"
- **Similarity is multi-faceted**: Two songs can be rhythmically similar but harmonically different, or vice versa
- **Machine learning reveals patterns humans miss**: The classifier identifies subtle spectral characteristics that distinguish genres beyond conscious perception

### Practical Lessons
- **Overfitting is the enemy**: With limited data (~300 songs), simpler models generalize better than complex deep networks
- **Visualization validates embeddings**: 2D/3D UMAP projections provide immediate visual confirmation that embeddings capture meaningful structure
- **Explainability builds trust**: SHAP explanations make the black-box model interpretable and debuggable

---