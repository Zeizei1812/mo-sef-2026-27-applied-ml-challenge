"""Étape 2 : comparaison de modèles sur un holdout aléatoire (10% du train)."""
import time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import features as F

df = pd.read_parquet("data/features.parquet")
tr = df[~df.is_test].reset_index(drop=True); del df
cols = F.feature_cols(tr)
rng = np.random.RandomState(42)
val = rng.rand(len(tr)) < 0.10
X, y = tr[cols], tr["label"].values
Xtr, Xva, ytr, yva = X[~val], X[val], y[~val], y[val]
print("n features", len(cols), "train", len(Xtr), "val", len(Xva))

# AUC univariée des features principales (contrôle de cohérence)
uni = {c: roc_auc_score(yva, Xva[c].fillna(Xva[c].median())) for c in cols}
uni = pd.Series(uni).map(lambda a: max(a, 1 - a)).sort_values(ascending=False)
print("Top AUC univariées:\n", uni.head(15).round(4).to_string())

res = {}
sub = rng.rand(len(Xtr)) < 0.2   # sous-échantillon pour les modèles lents
t = time.time()
lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=300))
lr.fit(Xtr[sub], ytr[sub]); res["LogisticRegression (20%)"] = roc_auc_score(yva, lr.predict_proba(Xva)[:, 1])
print(res, f"{time.time()-t:.0f}s")
t = time.time()
hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0)
hgb.fit(Xtr[sub], ytr[sub]); res["HistGradientBoosting (20%)"] = roc_auc_score(yva, hgb.predict_proba(Xva)[:, 1])
print(res, f"{time.time()-t:.0f}s")
t = time.time()
lg = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=127, verbose=-1, n_jobs=10)
lg.fit(Xtr[sub], ytr[sub]); res["LightGBM (20%)"] = roc_auc_score(yva, lg.predict_proba(Xva)[:, 1])
print(res, f"{time.time()-t:.0f}s")

# LightGBM complet avec early stopping
t = time.time()
params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=255,
              min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, max_bin=255, verbose=-1, num_threads=10)
dtr = lgb.Dataset(Xtr, ytr); dva = lgb.Dataset(Xva, yva, reference=dtr)
m = lgb.train(params, dtr, 3000, valid_sets=[dva],
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
res["LightGBM full (tuned)"] = m.best_score["valid_0"]["auc"]
print(f"{time.time()-t:.0f}s best_iter", m.best_iteration)
print("\nRésultats AUC validation:\n", pd.Series(res).round(5).to_string())
imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
print("\nTop importances (gain):\n", (imp / imp.sum()).head(25).round(4).to_string())
pd.Series({"best_iter": m.best_iteration, **res}).to_json("data/compare_results.json")
