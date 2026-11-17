"""Soumission rapide : LightGBM sur 20% du train (le modèle complet est dans train_final.py)."""
import numpy as np, pandas as pd, lightgbm as lgb, pyarrow.parquet as pq, pyarrow.compute as pc
import features as F

t = pq.read_table("data/features.parquet")
mask = pc.or_(t["is_test"], pc.less(pc.random(len(t)), 0.2))
df = t.filter(mask).to_pandas(); del t
cols = F.feature_cols(df)
tr, te = df[~df.is_test], df[df.is_test]
m = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=127, verbose=-1, num_threads=4),
              lgb.Dataset(tr[cols], tr["label"]), 300)
sub = pd.DataFrame({"id": te["id"].astype("int64").values, "label": m.predict(te[cols])}).sort_values("id")
assert len(sub) == 1006037 and sub.id.is_unique and sub.label.between(0, 1).all()
sub.to_csv("submission_quick.csv", index=False)
print("ok", len(sub), "train rows", len(tr))
