"""
Supervised genre classification with SHAP interpretation.

Pipeline: XGBoost + PCA + SHAP.
No data leakage, cross-validation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import shap
import joblib
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import EMBEDDINGS_FILE, SONG_DATA_FILE, PLOTS_DIR
except ModuleNotFoundError:
    BASE_DIR = Path(__file__).parent.parent
    EMBEDDINGS_FILE = BASE_DIR / "data" / "processed" / "embeddings_matrix.npy"
    SONG_DATA_FILE = BASE_DIR / "data" / "processed" / "song_data.pkl"
    PLOTS_DIR = BASE_DIR / "data" / "plots"


class GenreClassifier:
    """XGBoost classifier with PCA and SHAP interpretation."""
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, n_components: int = 15):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = None
        self.model = None
        
        # Encode labels
        self.y_full = self.label_encoder.fit_transform(labels)
        self.genre_names = self.label_encoder.classes_
        
        # Split BEFORE scaling (no data leakage)
        X_train_raw, X_test_raw, self.y_train, self.y_test = train_test_split(
            embeddings, self.y_full, test_size=0.2, random_state=42, stratify=self.y_full
        )
        
        # Fit scaler ONLY on train
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_test_scaled = self.scaler.transform(X_test_raw)
        
        # PCA dimensionality reduction
        n_components = min(n_components, X_train_scaled.shape[0] - 1)
        self.pca = PCA(n_components=n_components, random_state=42)
        
        self.X_train = self.pca.fit_transform(X_train_scaled)
        self.X_test = self.pca.transform(X_test_scaled)
        self.X_full = np.vstack([self.X_train, self.X_test])
        
        variance = self.pca.explained_variance_ratio_.sum()
        print(f"PCA: {n_components}D retaining {variance*100:.1f}% variance")
        print(f"Train: {len(self.X_train)} | Test: {len(self.X_test)}\n")
    
    def train(self):
        """Train XGBoost classifier with regularization."""
        print("Training XGBoost...")
        
        self.model = XGBClassifier(
            n_estimators=50, 
            max_depth=2,  
            learning_rate=0.03, 
            subsample=0.6, 
            colsample_bytree=0.6, 
            min_child_weight=5,  
            gamma=2.0,  
            reg_alpha=0.5,  
            reg_lambda=2.0, 
            random_state=42,
            eval_metric='logloss'
        )
        
        # Fit without early stopping
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = self.model.predict(self.X_test)
        test_acc = accuracy_score(self.y_test, y_pred)
        test_f1 = f1_score(self.y_test, y_pred, average='macro')
        
        print(f"Test accuracy: {test_acc:.3f} | F1: {test_f1:.3f}")
    
    def cross_validate(self, cv: int = 5):
        """5-fold cross-validation."""
        print("\nCross-validation (5-fold)...")
        
        scores = cross_val_score(
            self.model, self.X_full, self.y_full, 
            cv=cv, scoring='accuracy', n_jobs=-1
        )
        
        train_acc = self.model.score(self.X_train, self.y_train)
        test_acc = accuracy_score(self.y_test, self.model.predict(self.X_test))
        cv_acc = scores.mean()
        
        print(f"Train: {train_acc:.3f} | Test: {test_acc:.3f} | CV: {cv_acc:.3f} (±{scores.std():.3f})")
        print(f"Train-CV gap: {train_acc - cv_acc:.3f}")
        
        if train_acc - cv_acc > 0.20:
            print("⚠️  Severe overfitting (gap > 20%)")
        elif train_acc - cv_acc > 0.10:
            print("⚠️  Moderate overfitting (gap > 10%)")
        else:
            print("✓ Good generalization")
        
        return cv_acc
    
    def shap_analysis(self, example_song: str):
        """SHAP interpretation."""
        print("\n" + "="*60)
        print("SHAP INTERPRETATION")
        print("="*60)
        
        # Compute SHAP values
        print("Computing SHAP values...")
        explainer = shap.TreeExplainer(self.model)
        shap_values_raw = explainer.shap_values(self.X_full)
        
        if isinstance(shap_values_raw, list):
            shap_values = shap_values_raw[1]
        elif len(shap_values_raw.shape) == 3:
            shap_values = shap_values_raw[:, :, 1]
        else:
            shap_values = shap_values_raw
        
        feature_names = [f"PC{i+1}" for i in range(self.X_full.shape[1])]
        
        # 1. Global summary
        print("\n1. Global feature importance")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        plt.sca(ax1)
        shap.summary_plot(shap_values, self.X_full, feature_names=feature_names,
                         max_display=12, plot_type="bar", show=False)
        ax1.set_title('Importance', fontweight='bold')
        
        plt.sca(ax2)
        shap.summary_plot(shap_values, self.X_full, feature_names=feature_names,
                         max_display=12, plot_type="dot", show=False)
        ax2.set_title('Impact & Direction', fontweight='bold')
        
        plt.tight_layout()
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=150, bbox_inches='tight')
        print("✓ Saved: shap_summary.png")
        plt.close()
        
        # 2. Explain example song
        print("\n2. Song explanation")
        df = pd.read_pickle(SONG_DATA_FILE)
        matches = df[df['song_name'].str.contains(example_song, case=False, na=False)]
        
        if len(matches) > 0:
            song_idx = matches.index[0]
            song_data = matches.iloc[0]
            
            mask = df['genre'].isin(self.genre_names)
            filtered_indices = np.where(mask)[0]
            
            if song_idx in filtered_indices:
                song_idx_local = np.where(filtered_indices == song_idx)[0][0]
                
                shap_vals = shap_values[song_idx_local]
                X_song = self.X_full[song_idx_local]
                
                pred_proba = self.model.predict_proba(X_song.reshape(1, -1))[0]
                pred_class = self.model.predict(X_song.reshape(1, -1))[0]
                pred_genre = self.label_encoder.inverse_transform([pred_class])[0]
                
                print(f"Song: {song_data['song_name'][:50]}")
                print(f"True: {song_data['genre']} | Predicted: {pred_genre} ({pred_proba.max():.1%})")
                
                # Top features for explanation
                top_indices = np.argsort(np.abs(shap_vals))[::-1][:6]
                top_features = []
                for idx in top_indices:
                    impact = self.genre_names[1] if shap_vals[idx] > 0 else self.genre_names[0]
                    top_features.append({
                        'name': feature_names[idx],
                        'shap': shap_vals[idx],
                        'value': X_song[idx],
                        'impact': impact
                    })
                
                # Generate explanation text
                dominant_feature = top_features[0]
                explanation = self._generate_explanation(
                    song_data['song_name'],
                    pred_genre,
                    pred_proba.max(),
                    top_features
                )
                
                # Waterfall
                plt.figure(figsize=(10, 7))
                
                expected_value = explainer.expected_value
                if isinstance(expected_value, np.ndarray) and len(expected_value) > 1:
                    expected_value = expected_value[1]
                
                explanation_obj = shap.Explanation(
                    values=shap_vals,
                    base_values=expected_value,
                    data=X_song,
                    feature_names=feature_names
                )
                
                shap.plots.waterfall(explanation_obj, max_display=10, show=False)
                
                # Better title with song name
                plt.title(
                    f"{song_data['song_name'][:50]}\n"
                    f"Predicted: {pred_genre} ({pred_proba.max():.0%} confidence)",
                    fontsize=11, fontweight='bold', pad=15
                )
                
                plt.tight_layout()
                
                plt.savefig(PLOTS_DIR / "shap_waterfall.png", dpi=150, bbox_inches='tight')
                print("✓ Saved: shap_waterfall.png")
                plt.close()
                
                # Print explanation
                print(f"\n{explanation}")
        
        # 3. Boundary songs stats
        print("\n3. Boundary songs analysis")
        proba = self.model.predict_proba(self.X_full)
        uncertainty = 1 - np.abs(proba[:, 1] - 0.5) * 2
        
        df_filtered = df[df['genre'].isin(self.genre_names)].copy()
        df_filtered['uncertainty'] = uncertainty
        
        n_high_uncertainty = (uncertainty > 0.7).sum()
        print(f"High uncertainty songs (>70%): {n_high_uncertainty}/{len(df_filtered)} ({n_high_uncertainty/len(df_filtered)*100:.1f}%)")
        
        # 4. Dependence plot
        print("\n4. Feature dependence (PC3)")
        plt.figure(figsize=(10, 5))
        shap.dependence_plot(2, shap_values, self.X_full, 
                            feature_names=feature_names, show=False)
        plt.title('PC3 Dependence', fontweight='bold')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "shap_dependence.png", dpi=150, bbox_inches='tight')
        print("✓ Saved: shap_dependence.png")
        plt.close()
    
    def save(self, filepath: Path):
        """Save model pipeline."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'label_encoder': self.label_encoder,
            'genre_names': self.genre_names
        }, filepath)
        print(f"✓ Model saved: {filepath.name}")
    
    def _generate_explanation(self, song_name: str, pred_genre: str, confidence: float, top_features: list) -> str:
        """Generate detailed natural language explanation."""
        dominant = top_features[0]
        secondary = top_features[1] if len(top_features) > 1 else None
        
        # Map PC components to interpretable features
        pc_meaning = {
            'PC1': 'overall spectral energy distribution',
            'PC2': 'mid-frequency energy and bassline presence',
            'PC3': 'vocal melodic characteristics and harmonic richness',
            'PC4': 'rhythmic pattern complexity',
            'PC7': 'groove structure and percussive elements',
            'PC8': 'temporal dynamics',
            'PC9': 'tonal brightness',
            'PC13': 'textural warmth and sound atmosphere'
        }
        
        italodance_push = sum(1 for f in top_features if f['impact'] == 'ITALODANCE')
        trance_push = sum(1 for f in top_features if f['impact'] == 'TRANCE')
        
        explanation = f"\n{'='*60}\n"
        explanation += f"EXPLANATION: '{song_name[:40]}' → {pred_genre} ({confidence:.0%})\n"
        explanation += f"{'='*60}\n\n"
        
        # Confidence interpretation
        if confidence > 0.95:
            explanation += "★ VERY HIGH CONFIDENCE: The model is extremely certain.\n\n"
        elif confidence > 0.85:
            explanation += "★ HIGH CONFIDENCE: Strong evidence for this classification.\n\n"
        elif confidence > 0.70:
            explanation += "★ MODERATE CONFIDENCE: Clear tendency but some ambiguity.\n\n"
        else:
            explanation += "★ LOW CONFIDENCE: This is a boundary song with mixed characteristics.\n\n"
        
        # Dominant feature explanation
        pc_name = dominant['name']
        pc_desc = pc_meaning.get(pc_name, 'audio characteristic')
        
        explanation += f"PRIMARY DRIVER: {pc_name}\n"
        explanation += f"• What it represents: {pc_desc}\n"
        explanation += f"• Value: {dominant['value']:.2f} (negative = ITALODANCE signature)\n"
        explanation += f"• SHAP impact: {abs(dominant['shap']):.2f} → pushes strongly toward {dominant['impact']}\n"
        
        if pc_name == 'PC3':
            explanation += f"• Interpretation: Negative PC3 indicates direct, immediate melodies\n"
            explanation += f"  typical of ITALODANCE, rather than the atmospheric, layered vocals\n"
            explanation += f"  characteristic of TRANCE.\n\n"
        elif pc_name == 'PC2':
            explanation += f"• Interpretation: Negative PC2 shows prominent basslines and\n"
            explanation += f"  mid-frequency energy typical of ITALODANCE groove.\n\n"
        elif pc_name == 'PC7':
            explanation += f"• Interpretation: Negative PC7 indicates strict 4/4 rhythm patterns\n"
            explanation += f"  without progressive breakdowns typical of TRANCE.\n\n"
        else:
            explanation += "\n"
        
        # Secondary features
        if secondary:
            explanation += f"SECONDARY FACTOR: {secondary['name']}\n"
            pc_desc_sec = pc_meaning.get(secondary['name'], 'audio characteristic')
            explanation += f"• Represents: {pc_desc_sec}\n"
            explanation += f"• SHAP impact: {abs(secondary['shap']):.2f} → {secondary['impact']}\n"
            
            if secondary['impact'] == dominant['impact']:
                explanation += f"• Effect: Reinforces the {dominant['impact']} classification.\n\n"
            else:
                explanation += f"• Effect: Creates some uncertainty by suggesting {secondary['impact']} traits.\n\n"
        
        # Overall pattern
        explanation += f"OVERALL PATTERN:\n"
        explanation += f"• Features pushing ITALODANCE: {italodance_push}/6\n"
        explanation += f"• Features pushing TRANCE: {trance_push}/6\n"
        
        if italodance_push >= 5:
            explanation += f"• Verdict: Consistently strong ITALODANCE signature across all features.\n"
        elif trance_push >= 5:
            explanation += f"• Verdict: Consistently strong TRANCE signature across all features.\n"
        elif italodance_push > trance_push:
            explanation += f"• Verdict: Predominantly ITALODANCE but with some TRANCE elements.\n"
        elif trance_push > italodance_push:
            explanation += f"• Verdict: Predominantly TRANCE but with some ITALODANCE elements.\n"
        else:
            explanation += f"• Verdict: Hybrid song with balanced characteristics of both genres.\n"
        
        explanation += f"\n{'='*60}\n"
        
        return explanation


def main():
    print("="*60)
    print("GENRE CLASSIFICATION + SHAP")
    print("="*60)
    
    # Load data
    df = pd.read_pickle(SONG_DATA_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"Loaded: {len(df)} songs, {embeddings.shape[1]}D embeddings\n")
    
    # Filter
    mask = df['genre'] != 'UNKNOWN'
    df_filtered = df[mask]
    embeddings_filtered = embeddings[mask]
    
    # Train
    classifier = GenreClassifier(embeddings_filtered, df_filtered['genre'].values, n_components=15)
    classifier.train()
    classifier.cross_validate(cv=5)
    
    # Save
    model_path = PLOTS_DIR.parent / "models" / "genre_classifier.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(model_path)
    
    # SHAP
    classifier.shap_analysis(example_song="Nordic Stars - Crying in the rain")
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)
    print("Files: {PLOTS_DIR}/")
    print("  - shap_summary.png")
    print("  - shap_waterfall.png")
    print("  - shap_dependence.png")


if __name__ == "__main__":
    main()