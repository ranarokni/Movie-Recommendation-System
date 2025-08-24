import pandas as pd
from typing import Iterable, Optional, List, Tuple
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

class CollaborativeRecommender:
    """
    Ratings-only wrapper around Surprise.
    Expects a DataFrame with columns: ['userId','movieId','rating'].
    Returns movieIds (and estimated scores if requested).
    """

    def __init__(self, algo=None, rating_scale: Optional[Tuple[float, float]] = None, random_state: int = 42):
        self.algo = algo if algo is not None else SVD(random_state=random_state)
        self.rating_scale = rating_scale  # infer from data if None
        self.reader = None
        self.trainset = None
        self._ratings_df = None
        self._user_seen = {}  # userId -> set(movieId)

    def fit(self, ratings: pd.DataFrame):
        """Fit on full ratings set."""
        req = {"userId","movieId","rating"}
        if not req.issubset(ratings.columns):
            raise ValueError(f"ratings must have columns {sorted(req)}")

        self._ratings_df = ratings.copy()

        if self.rating_scale is None:
            self.rating_scale = (float(ratings["rating"].min()), float(ratings["rating"].max()))

        self.reader = Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(self._ratings_df[["userId","movieId","rating"]], self.reader)

        self.trainset = data.build_full_trainset()
        self.algo.fit(self.trainset)

        # cache items each user has rated
        self._user_seen = (
            self._ratings_df.groupby("userId")["movieId"]
            .apply(lambda s: set(s.tolist()))
            .to_dict()
        )
        return self

    def cross_validate(self, cv: int = 5, measures: List[str] = ["RMSE","MAE"], verbose: bool = False):
        """5-fold CV using the current algorithm configuration."""
        if self._ratings_df is None:
            raise RuntimeError("Call fit(...) first to provide ratings.")
        data = Dataset.load_from_df(self._ratings_df[["userId","movieId","rating"]],
                                    Reader(rating_scale=self.rating_scale))
        return cross_validate(self.algo, data, measures=measures, cv=cv, verbose=verbose, n_jobs=1)

    def predict(self, user_id, movie_id):
        """One-off predicted rating (Surprise Prediction object)."""
        if self.trainset is None:
            raise RuntimeError("Call fit(...) before predict().")
        return self.algo.predict(uid=user_id, iid=movie_id, r_ui=None, verbose=False)

    def _anti_items_for_user(self, user_id, restrict_to: Optional[Iterable] = None):
        """Items the user hasn't rated yet (as raw movieIds)."""
        all_items = set(self._ratings_df["movieId"].unique()) if restrict_to is None else set(restrict_to)
        seen = self._user_seen.get(user_id, set())
        return list(all_items - seen)

    def recommend(self, user_id, k: int = 10, filter_seen: bool = True,
                  candidates: Optional[Iterable] = None, include_scores: bool = False):
        """
        Top-k movieIds for a user.
        - filter_seen: exclude already-rated items
        - candidates: optional iterable of movieIds to score (e.g., only popular)
        - include_scores: include estimated ratings
        """
        if self.trainset is None:
            raise RuntimeError("Call fit(...) before recommend().")

        to_score = (self._anti_items_for_user(user_id, candidates) if filter_seen
                    else list(set(self._ratings_df["movieId"].unique()) if candidates is None else set(candidates)))

        if not to_score:
            return (pd.DataFrame(columns=["movieId","est"]) if include_scores
                    else pd.Series(dtype=int, name="movieId"))

        preds = [self.algo.predict(user_id, iid) for iid in to_score]
        preds.sort(key=lambda p: p.est, reverse=True)
        top = preds[:k]

        mids = [p.iid for p in top]
        if include_scores:
            ests = [round(p.est, 4) for p in top]
            return pd.DataFrame({"movieId": mids, "est": ests})

        return pd.Series(mids, name="movieId")

    def recommend_for_users(self, user_ids: Iterable, k: int = 10, include_scores: bool = False):
        """Batch convenience: dict[user_id -> Series/DataFrame]."""
        return {uid: self.recommend(uid, k=k, include_scores=include_scores) for uid in user_ids}
