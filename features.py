"""Nettoyage + feature engineering, appliqués conjointement à train et test.
import warnings; warnings.filterwarnings("ignore", category=__import__("pandas").errors.PerformanceWarning)

Idée centrale : le test est un échantillon aléatoire (~10%) des mêmes sessions /
utilisateurs / période que le train. Pour chaque ligne, on peut donc utiliser les
labels des AUTRES lignes labellisées (leave-one-out). Une ligne de train et une
ligne de test sont alors dans une situation statistiquement identique : leurs
features ne contiennent jamais leur propre label.
"""
import numpy as np
import pandas as pd

CAT_COLS = ["category_l1", "category_l2", "category_l3", "brand", "condition",
            "size", "shipping_method"]


def load():
    tr = pd.read_parquet("data/train.parquet")
    te = pd.read_parquet("data/test.parquet")
    tr["id"] = -1
    te["label"] = np.nan
    df = pd.concat([tr, te], ignore_index=True)
    df["is_test"] = df["label"].isna()
    return df


# --------------------------------------------------------------------- nettoyage
def _mode(df, keys, val):
    """Valeur la plus fréquente de `val` par groupe `keys` (vectorisé)."""
    c = df.dropna(subset=keys + [val]).groupby(keys + [val]).size().reset_index(name="n")
    c = c.sort_values("n", ascending=False).drop_duplicates(keys)
    return c.set_index(keys)[val]


def clean(df):
    # Indicateurs de valeurs manquantes (porteurs de signal, cf. EDA)
    for c in ["user_id", "item_id", "session_id", "sequence_id", "size",
              "category_l2", "category_l3"]:
        df[f"na_{c}"] = df[c].isna().astype("int8")

    # user_id manquant -> utilisateur majoritaire de la session
    sess_user = _mode(df, ["session_id"], "user_id")
    m = df["user_id"].isna() & df["session_id"].notna()
    df.loc[m, "user_id"] = df.loc[m, "session_id"].map(sess_user)

    # session_id manquant -> session de l'utilisateur la plus proche dans le temps
    # (approximation : même user + même sequence_id)
    key = ["user_id", "sequence_id"]
    us = _mode(df, key, "session_id")
    m = df["session_id"].isna() & df["user_id"].notna() & df["sequence_id"].notna()
    idx = pd.MultiIndex.from_frame(df.loc[m, key])
    df.loc[m, "session_id"] = us.reindex(idx).values

    # item_id manquant -> item ayant exactement la même signature d'attributs
    sig_cols = ["price", "category_l3", "brand", "condition", "size", "shipping_method"]
    df["item_sig"] = pd.util.hash_pandas_object(df[sig_cols].fillna(-1), index=False).values
    sig_item = _mode(df, ["item_sig"], "item_id")
    m = df["item_id"].isna()
    df.loc[m, "item_id"] = df.loc[m, "item_sig"].map(sig_item)

    for c in ["user_id", "item_id", "session_id"]:
        df[f"rec_{c}"] = (df[f"na_{c}"].astype(bool) & df[c].notna()).astype("int8")

    # Catégories manquantes -> -1 (LightGBM gère, mais on garde des entiers)
    for c in CAT_COLS:
        df[c] = df[c].fillna(-1).astype("int32")
    return df


# --------------------------------------------------------------------- features
N_FOLDS = 10


def _oof(df, keys, name, prior, alpha=5.0):
    """Encodage cible out-of-fold.

    Ligne de train du fold k : moyenne lissée des labels du groupe calculée sur les
    lignes de train des AUTRES folds (son propre label et ceux de son fold sont exclus).
    Ligne de test : calculée sur tout le train.
    NB : un leave-one-out simple fuit le label (valeur différente selon y=0/1 dans un
    même groupe) -> AUC=1 en validation. Le K-fold évite cela.
    """
    valid = df[keys].notna().all(axis=1)
    sub = df.loc[valid, keys + ["fold", "label"]]
    tot = sub.groupby(keys, sort=False)["label"].agg(["sum", "count"])
    per = sub[sub.fold >= 0].groupby(keys + ["fold"], sort=False)["label"].agg(["sum", "count"])
    kidx = pd.MultiIndex.from_frame(sub[keys]) if len(keys) > 1 else pd.Index(sub[keys[0]])
    S = tot["sum"].reindex(kidx).values
    N = tot["count"].reindex(kidx).values
    fidx = pd.MultiIndex.from_frame(sub[keys + ["fold"]])
    Sf = np.nan_to_num(per["sum"].reindex(fidx).values)
    Nf = np.nan_to_num(per["count"].reindex(fidx).values)
    s_o, n_o = S - Sf, N - Nf
    m = valid.values
    te = np.full(len(df), np.nan, dtype="float32")
    n = np.full(len(df), np.nan, dtype="float32")
    te[m] = (s_o + alpha * prior) / (n_o + alpha)
    n[m] = n_o
    df[f"te_{name}"] = te
    df[f"n_{name}"] = n


def build(df):
    prior = df["label"].mean()
    rng = np.random.RandomState(0)
    df["fold"] = np.where(df["is_test"], -1, rng.randint(0, N_FOLDS, len(df))).astype("int8")
    t = df["event_time"]
    df["ts"] = (df["event_time"] - pd.Timestamp("2023-01-01")).dt.total_seconds().astype("float64")
    df["hour"] = t.dt.hour.astype("int8")
    df["dow"] = t.dt.dayofweek.astype("int8")
    df["month"] = t.dt.month.astype("int8")
    df["log_price"] = np.log1p(df["price"]).astype("float32")

    # --- prix relatif
    for c in ["category_l3", "category_l2", "brand"]:
        med = df.groupby(c)["log_price"].transform("median")
        df[f"lp_rel_{c}"] = (df["log_price"] - med).astype("float32")

    # --- statistiques leave-one-out des labels
    combos = {
        "sess": ["session_id"],
        "sessseq": ["session_id", "sequence_id"],
        "user": ["user_id"],
        "item": ["item_id"],
        "brand": ["brand"],
        "cat3": ["category_l3"],
        "cat2": ["category_l2"],
        "size": ["size"],
        "user_cat1": ["user_id", "category_l1"],
        "user_cat3": ["user_id", "category_l3"],
        "user_brand": ["user_id", "brand"],
        "user_day": ["user_id", "month", "dow"],
        "seq": ["sequence_id"],
        "seqlen": ["sequence_length"],
    }
    for name, keys in combos.items():
        _oof(df, keys, name, prior)

    # --- contexte temporel dans la session (trié par temps)
    df["_s"] = df["session_id"].fillna(-df.index.to_series() - 1)
    df.sort_values(["_s", "ts"], inplace=True)
    g = df.groupby("_s", sort=False)
    df["pos_in_sess"] = g.cumcount().astype("int32")
    df["sess_len"] = g["ts"].transform("size").astype("int32")
    df["pos_rel"] = (df["pos_in_sess"] / df["sess_len"]).astype("float32")
    df["dt_prev"] = g["ts"].diff().astype("float32")
    df["dt_next"] = (-g["ts"].diff(-1)).astype("float32")
    df["dt_prev2"] = g["ts"].diff(2).astype("float32")
    df["dt_next2"] = (-g["ts"].diff(-2)).astype("float32")
    df["sess_dur"] = (g["ts"].transform("max") - g["ts"].transform("min")).astype("float32")
    df["dt_from_start"] = (df["ts"] - g["ts"].transform("min")).astype("float32")
    df["lp_rel_sess"] = (df["log_price"] - g["log_price"].transform("mean")).astype("float32")
    df["same_cat3_prev"] = (df["category_l3"] == g["category_l3"].shift(1)).astype("int8")
    df["same_cat3_next"] = (df["category_l3"] == g["category_l3"].shift(-1)).astype("int8")
    for k in [1, 2, 3, 5]:
        df[f"lab_prev{k}"] = g["label"].shift(k).astype("float32")
        df[f"lab_next{k}"] = g["label"].shift(-k).astype("float32")
    # moyenne des labels connus dans une fenêtre ±5 (hors ligne courante)
    for w in [3, 10]:
        sy = sum(g["label"].shift(k).fillna(0) + g["label"].shift(-k).fillna(0)
                 for k in range(1, w + 1))
        sn = sum(g["label"].shift(k).notna().astype("float32") + g["label"].shift(-k).notna().astype("float32")
                 for k in range(1, w + 1))
        df[f"win{w}_mean"] = (sy / sn.replace(0, np.nan)).astype("float32")
        df[f"win{w}_n"] = sn.astype("float32")

    # --- contexte temporel utilisateur (toutes sessions)
    df["_u"] = df["user_id"].fillna(-df.index.to_series() - 1)
    df.sort_values(["_u", "ts"], inplace=True)
    gu = df.groupby("_u", sort=False)
    df["u_dt_prev"] = gu["ts"].diff().astype("float32")
    df["u_dt_next"] = (-gu["ts"].diff(-1)).astype("float32")
    df["u_lab_prev1"] = gu["label"].shift(1).astype("float32")
    df["u_lab_next1"] = gu["label"].shift(-1).astype("float32")
    df["u_n_sessions"] = gu["session_id"].transform("nunique").astype("float32")

    # --- popularité (train+test, sans label)
    for c in ["item_id", "brand", "category_l3", "user_id"]:
        df[f"pop_{c}"] = df.groupby(c)["ts"].transform("size").astype("float32")
    df["item_n_users"] = df.groupby("item_id")["user_id"].transform("nunique").astype("float32")
    df["user_n_items"] = df.groupby("user_id")["item_id"].transform("nunique").astype("float32")
    df["user_item_cnt"] = df.groupby(["user_id", "item_id"])["ts"].transform("size").astype("float32")

    df.sort_index(inplace=True)
    df.drop(columns=["_s", "_u"], inplace=True)
    return df


DROP = ["label", "id", "fold", "is_test", "event_time", "ts", "item_sig",
        "user_id", "item_id", "session_id"]


def feature_cols(df):
    return [c for c in df.columns if c not in DROP]
