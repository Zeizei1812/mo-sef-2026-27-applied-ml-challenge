"""Étape 1 : nettoyage + features -> data/features.parquet"""
import time
import features as F

t0 = time.time()
df = F.load()
df = F.clean(df)
print(f"clean ok {time.time()-t0:.0f}s")
for c in ["user_id", "item_id", "session_id"]:
    print(c, "NA restant train/test:", df.loc[~df.is_test, c].isna().mean().round(4),
          df.loc[df.is_test, c].isna().mean().round(4), "récupérés:", int(df[f"rec_{c}"].sum()))
df = F.build(df)
print(f"build ok {time.time()-t0:.0f}s, shape {df.shape}")
df.to_parquet("data/features.parquet")
