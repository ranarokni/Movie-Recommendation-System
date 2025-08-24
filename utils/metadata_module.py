import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from collections.abc import Iterable

class MetadataIndexer:
    """
    Build a CountVectorizer content index over movie metadata "soup"
    (e.g., keywords, cast, director, genres, language, etc.) and
    return top-K similar titles given 1+ seed titles.
    """

    def __init__(
        self,
        meta_cols=("keywords", "cast", "director", "genres", "original_language"),
        ngram_range=(1, 2),
        min_df=1,                 
        max_features=None,
        stop_words="english",
        weights=None,             
    ):
        self.meta_cols = meta_cols
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.stop_words = stop_words
        self.weights = weights or {}

        self.vectorizer = None
        self.matrix = None
        self.df = None
        self.titles = None
        self.indices = None

    def _ensure_tokens(self, value):
        """Turn a cell into a list[str] tokens."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            out = []
            for v in value:
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    out.append(s)
            return out
        s = str(value).strip()
        if not s:
            return []
        return s.split()

    def _build_soup(self, df: pd.DataFrame) -> pd.Series:
        cols = [c for c in self.meta_cols if c in df.columns]
        if not cols:
            raise ValueError(f"No valid metadata columns found among {self.meta_cols}")

        def row_to_soup(row):
            tokens = []
            for c in cols:
                toks = self._ensure_tokens(row[c])
                w = float(self.weights.get(c, 1.0))
                if w <= 0 or not toks:
                    continue
                # simple weighting by repetition (integer part)
                reps = max(1, int(round(w)))
                for _ in range(reps):
                    tokens.extend(toks)
            return " ".join(tokens) if tokens else ""

        return df.apply(row_to_soup, axis=1)


    def fit(self, df: pd.DataFrame):
        """
        Expects a DataFrame with at least: 'title' + the chosen meta_cols.
        """
        if "title" not in df.columns:
            raise ValueError("DataFrame must contain a 'title' column.")

        self.df = df.reset_index(drop=True).copy()
        self.titles = self.df["title"]
        # If duplicate titles exist, keep the first index for mapping
        self.indices = pd.Series(self.df.index, index=self.titles).drop_duplicates()

        soup = self._build_soup(self.df)

        self.vectorizer = CountVectorizer(
            analyzer="word",
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            stop_words=self.stop_words,
        )
        self.matrix = self.vectorizer.fit_transform(soup)
        return self

    def _scores_for_index(self, idx: int) -> np.ndarray:
        sims = linear_kernel(self.matrix[idx], self.matrix).ravel()
        return sims

    def recommend(self, titles, k=10, include_scores=False):
        """
        Recommend top-K titles similar to one or more seed titles.
        """
        if self.indices is None or self.matrix is None:
            raise RuntimeError("Call fit(df) before recommend().")

        if isinstance(titles, str):
            titles = [titles]

        missing = [t for t in titles if t not in self.indices]
        if missing:
            raise KeyError(f"Titles not found: {missing}")

        idxs = [int(self.indices[t]) for t in titles]
        seed_vecs = self.matrix[idxs]

        profile = seed_vecs.sum(axis=0) / len(idxs)
        profile = csr_matrix(profile)
        profile = normalize(profile)

        sims = linear_kernel(profile, self.matrix).ravel()

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
        """
        if self.matrix is None:
            raise RuntimeError("Call fit(df) before recommend_by_index().")
        if idx < 0 or idx >= self.matrix.shape[0]:
            raise IndexError(f"idx must be in [0, {self.matrix.shape[0]-1}]")

        sims = self._scores_for_index(idx)
        order = np.argsort(-sims)
        order = [i for i in order if i != idx]
        top_idx = order[:k]
        top_titles = self.titles.iloc[top_idx].reset_index(drop=True)

        if include_scores:
            top_scores = pd.Series(sims[top_idx]).reset_index(drop=True).round(6)
            return pd.DataFrame({"title": top_titles, "score": top_scores})

        return top_titles
