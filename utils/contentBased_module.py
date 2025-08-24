from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd


class ContentIndexer2:
    """
    Build a TF-IDF content index over movie descriptions and
    return top-K similar titles given one input title.
    """
    def __init__(
        self,
        ngram_range=(1, 2),
        min_df=1,                 
        max_features=None,
        stop_words="english",
        text_cols=("overview", "tagline")
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.stop_words = stop_words
        self.text_cols = text_cols

        self.vectorizer = None
        self.tfidf = None
        self.df = None
        self.titles = None
        self.indices = None

    def _build_text(self, df: pd.DataFrame) -> pd.Series:
        cols = [c for c in self.text_cols if c in df.columns]
        parts = [df[c].fillna('') for c in cols]
        if not parts:
            raise ValueError(f"No valid text columns found among {self.text_cols}")
        desc = parts[0]
        for p in parts[1:]:
            desc = desc + " " + p
        return desc.fillna('')

    def fit(self, df: pd.DataFrame):
        """
        Expects a DataFrame with at least: 'title' plus text_cols (overview/tagline).
        """
        if "title" not in df.columns:
            raise ValueError("DataFrame must contain a 'title' column.")

        self.df = df.reset_index(drop=True).copy()
        self.titles = self.df["title"]

        self.indices = pd.Series(self.df.index, index=self.titles).drop_duplicates()

        desc = self._build_text(self.df)

        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            stop_words=self.stop_words
        )
        self.tfidf = self.vectorizer.fit_transform(desc)
        return self

    def _scores_for_index(self, idx: int) -> np.ndarray:
        sims = linear_kernel(self.tfidf[idx], self.tfidf).ravel()
        return sims

    def recommend(self, titles, k=10, include_scores=False):
        if self.indices is None or self.tfidf is None:
            raise RuntimeError("Call fit(df) before recommend().")
    
        if isinstance(titles, str):
            titles = [titles]
    
        missing = [t for t in titles if t not in self.indices]
        if missing:
            raise KeyError(f"Titles not found: {missing}")
    
        idxs = [int(self.indices[t]) for t in titles]
        seed_vecs = self.tfidf[idxs]
    
        profile = seed_vecs.sum(axis=0) / len(idxs)   
        profile = csr_matrix(profile)                 
        profile = normalize(profile)                    
    
        sims = linear_kernel(profile, self.tfidf).ravel()
    
        order = np.argsort(-sims)
        exclude_idx = set(idxs)
        order = [i for i in order if i not in exclude_idx]
    
        top_idx = order[:k]
        top_titles = self.titles.iloc[top_idx].reset_index(drop=True)
    
        if include_scores:
            top_scores = pd.Series(sims[top_idx]).reset_index(drop=True).round(6)
            return pd.DataFrame({"title": top_titles, "score": top_scores})
    
        return top_titles

    def recommend_by_index(self, idx: int, k: int = 10, include_scores: bool = False):
        """
        Same as recommend(), but starting from a row index instead of a title.
        Useful if you already looked up the index elsewhere.
        """
        if idx < 0 or idx >= self.tfidf.shape[0]:
            raise IndexError(f"idx must be in [0, {self.tfidf.shape[0]-1}]")
        sims = self._scores_for_index(idx)
        order = np.argsort(-sims)
        order = order[order != idx]
        top_idx = order[:k]
        top_titles = self.titles.iloc[top_idx].reset_index(drop=True)
        if include_scores:
            top_scores = pd.Series(sims[top_idx]).reset_index(drop=True).round(6)
            return pd.DataFrame({"title": top_titles, "score": top_scores})
        return top_titles
