"""Étape 3 : LightGBM -> early stopping sur holdout 10%, puis réentraînement sur tout le train
-> submission.csv"""
import numpy as np
import lightgbm as lgb
import pandas as pd
import features as F

df = pd.read_parquet("data/features.parquet")
cols = F.feature_cols(df)
tr, te = df[~df.is_test], df[df.is_test]
del df

params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=255,
              min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, max_bin=255, verbose=-1, num_threads=10)

# 1) holdout aléatoire de 10% pour trouver le nombre d'arbres et mesurer l'AUC
val = np.random.RandomState(42).rand(len(tr)) < 0.10
dtr = lgb.Dataset(tr.loc[~val, cols], tr.loc[~val, "label"])
dva = lgb.Dataset(tr.loc[val, cols], tr.loc[val, "label"], reference=dtr)
m = lgb.train(params, dtr, 3000, valid_sets=[dva],
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
print(f"AUC validation = {m.best_score['valid_0']['auc']:.5f} @ {m.best_iteration} arbres", flush=True)
imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
print((imp / imp.sum()).head(20).round(4).to_string(), flush=True)
n_iter = int(m.best_iteration * 1.1)  # +10% de données -> un peu plus d'arbres
del dtr, dva, m

# 2) réentraînement sur 100% du train, moyenne de seeds
preds = 0
seeds = [0, 1, 2]
for s in seeds:  # moyenne de 3 seeds pour stabiliser
    m = lgb.train({**params, "seed": s}, lgb.Dataset(tr[cols], tr["label"]), n_iter)
    preds = preds + m.predict(te[cols]) / len(seeds)
    print("seed", s, "ok", flush=True)

sub = pd.DataFrame({"id": te["id"].astype("int64").values, "label": preds})
sub = sub.sort_values("id")
assert len(sub) == 1006037 and sub.id.is_unique and sub.label.between(0, 1).all()
sub.to_csv("submission.csv", index=False)
print(sub.describe())
