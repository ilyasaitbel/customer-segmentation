from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "customer_segmentation_model.pkl"

FEATURES = ["Recence", "Frequence", "Montant"]


def find_dataset():
    possible_names = [
        "data.xlsx",
        "Online Retail.xlsx",
        "online_retail.xlsx",
        "online_retail.csv",
        "data.csv",
    ]

    for name in possible_names:
        path = DATA_DIR / name
        if path.exists():
            return path

    files = list(DATA_DIR.glob("*.xlsx")) + list(DATA_DIR.glob("*.csv"))
    if files:
        return files[0]
    return None


@st.cache_data
def load_data(path):
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path, encoding="latin1")
    return pd.read_excel(path)


def prepare_transactions(df):
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=["CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_rfm(df):
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recence=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequence=("InvoiceNo", "nunique"),
        Montant=("TotalPrice", "sum"),
    )

    return rfm.reset_index()


def name_segments(profile):
    profile = profile.copy()

    score = (
        profile["Montant"].rank(ascending=True)
        + profile["Frequence"].rank(ascending=True)
        + profile["Recence"].rank(ascending=False)
    )

    ordered_clusters = score.sort_values().index.tolist()
    return {
        ordered_clusters[0]: "Clients occasionnels / inactifs",
        ordered_clusters[1]: "Clients intermediaires",
        ordered_clusters[2]: "Clients a forte valeur",
    }


def train_models(rfm):
    model_data = rfm.copy()

    # Log reduces the effect of very large values before K-Means.
    x_log = np.log1p(model_data[FEATURES])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_log)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    model_data["Cluster"] = kmeans.fit_predict(x_scaled)

    profile = model_data.groupby("Cluster")[FEATURES].mean()
    segment_names = name_segments(profile)
    model_data["Segment"] = model_data["Cluster"].map(segment_names)

    x = model_data[FEATURES]
    y = model_data["Cluster"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(x_train, y_train)

    return {
        "rf_model": rf,
        "kmeans_model": kmeans,
        "scaler": scaler,
        "segment_names": segment_names,
        "profile": profile,
        "rfm": model_data,
    }


def load_or_train_bundle():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH), "loaded"

    dataset_path = find_dataset()
    if dataset_path is None:
        return None, "missing_data"

    df = load_data(dataset_path)
    clean_df = prepare_transactions(df)
    rfm = build_rfm(clean_df)
    bundle = train_models(rfm)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle, "trained"


def predict_segment(bundle, recence, frequence, montant):
    new_client = pd.DataFrame(
        [[recence, frequence, montant]],
        columns=FEATURES,
    )
    cluster = int(bundle["rf_model"].predict(new_client)[0])
    segment = bundle["segment_names"].get(cluster, f"Cluster {cluster}")
    return cluster, segment, new_client


st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("Segmentation des clients")
st.write("Entrez les indicateurs RFM d'un client pour predire son segment.")

bundle, status = load_or_train_bundle()

if status == "missing_data":
    st.error("Dataset introuvable.")
    st.info("Ajoutez le fichier Online Retail dans le dossier data avec un nom comme data.xlsx ou Online Retail.xlsx.")
    st.stop()

if status == "trained":
    st.success("Modele entraine et sauvegarde dans le dossier models.")
else:
    st.caption("Modele charge depuis le dossier models.")

recence = st.number_input("Recence (jours depuis le dernier achat)", min_value=0, value=30, step=1)
frequence = st.number_input("Frequence (nombre de factures)", min_value=1, value=5, step=1)
montant = st.number_input("Montant total depense", min_value=0.0, value=500.0, step=10.0)

if st.button("Predire le segment", type="primary"):
    cluster, segment, client = predict_segment(bundle, recence, frequence, montant)

    st.subheader("Resultat")
    st.metric("Cluster predit", cluster)
    st.success(segment)
    st.dataframe(client, use_container_width=True)

st.subheader("Profil moyen des segments")
profile = bundle["profile"].copy()
profile["Segment"] = profile.index.map(bundle["segment_names"])
st.dataframe(profile.reset_index(), use_container_width=True)

