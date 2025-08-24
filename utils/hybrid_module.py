import pandas as pd
import numpy as np
from typing import Optional

class HybridRecommender:
    """
    Re-rank content-similar candidates with collaborative SVD predictions.

    - content: FITTED MetadataIndexer (has .indices, .titles, .df with 'id')
    - collab:  FITTED CollaborativeRecommenderRO (has .predict)
    - id_map:  DataFrame with columns ['movieId','id'] (TMDb id), can have duplicate titles
               You can build it like you did, but we won't index by title anymore.
    - smd (optional): metadata DF with ['title','vote_count','vote_average','year','id']
    """

    def __init__(self, content, collab, id_map: pd.DataFrame, smd: Optional[pd.DataFrame] = None):
        self.content = content
        self.collab = collab

        need = {"movieId", "id"}
        if not need.issubset(id_map.columns):
            raise ValueError(f"id_map must include columns {sorted(need)}")
        self.id_map = id_map[["movieId","id"]].dropna().copy()
        # id -> movieId for fast lookup (may still be many-to-one but that's fine)
        self.indices_map_by_id = self.id_map.drop_duplicates("id").set_index("id")

        self.smd = smd

    def recommend(self, user_id, title: str, k: int = 10, n_content: int = 200,
              include_meta: bool = True, alpha: float = 0.5, genre_filter: list[str] | None = None):
        # 1) content neighbors (get sims + titles)
        if title not in self.content.indices:
            raise KeyError(f"Title not found in content index: {title}")
        seed_idx = int(self.content.indices[title])
        sims_vec = self.content._scores_for_index(seed_idx)
        order = np.argsort(-sims_vec)
        order = [i for i in order if i != seed_idx]
        cand_idx = order[:n_content]
        cand_titles = self.content.titles.iloc[cand_idx].tolist()
        cand_sims = sims_vec[cand_idx]
    
        # 2) map by TMDb id -> movieId (no duplicate-title issues)
        if "id" not in self.content.df.columns:
            raise ValueError("content.df must have an 'id' (TMDb) column for hybrid mapping.")
        cand_ids = self.content.df.iloc[cand_idx]["id"].astype(int).values
        cand_df = pd.DataFrame({"title": cand_titles, "id": cand_ids, "sim": cand_sims})
    
        mapped = cand_df.merge(self.indices_map_by_id[["movieId"]].reset_index(),
                               on="id", how="left").dropna(subset=["movieId"]).drop_duplicates("movieId")
        if mapped.empty:
            return pd.DataFrame(columns=["title","movieId","id","sim","est","score"])
    
        mapped["movieId"] = mapped["movieId"].astype(int)
    
        if genre_filter and self.smd is not None and "genres" in self.smd.columns:
            gmap = self.smd[["id","genres"]].drop_duplicates("id")
            mapped = mapped.merge(gmap, on="id", how="left")
            def has_any(gs): 
                return any(g in set(gs or []) for g in genre_filter)
            mapped = mapped[mapped["genres"].apply(has_any)]
            mapped = mapped.drop(columns=["genres"])
    
        # 3) SVD estimate
        mapped["est"] = mapped["movieId"].apply(lambda mid: self.collab.predict(user_id, int(mid)).est)
    
        # 4) BLEND: normalize sim & est to [0,1], then combine
        s = mapped["sim"].to_numpy()
        e = mapped["est"].to_numpy()
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        e = (e - e.min()) / (e.max() - e.min() + 1e-9)
        mapped["score"] = alpha * s + (1 - alpha) * e
    
        # 5) pretty join (optional) and return
        out = mapped.copy()
        if include_meta and self.smd is not None:
            keep = [c for c in ["id","title","vote_count","vote_average","year"] if c in self.smd.columns]
            if keep:
                out = out.merge(self.smd[keep].drop_duplicates("id"), on=["id","title"], how="left")
    
        out = out.sort_values("score", ascending=False).head(k)
        cols = [c for c in ["title","movieId","id","sim","est","score","vote_count","vote_average","year"] if c in out.columns]
        return out[cols].reset_index(drop=True)
