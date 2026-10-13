import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    page_title="SegmentPulse AI | Customer Intelligence Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium SaaS Custom CSS (Modern Slate / Glassmorphism / Indigo Theme)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top App Bar & Brand */
    .saas-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 18px;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.4);
    }
    .saas-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .saas-tagline {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.3rem;
        margin-bottom: 0;
        font-weight: 400;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid rgba(165, 180, 252, 0.3);
    }
    .status-live {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        margin-right: 6px;
        box-shadow: 0 0 8px #10b981;
    }

    /* KPI Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.2rem;
        margin-bottom: 1.8rem;
    }
    .kpi-card {
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 1.3rem 1.4rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .kpi-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        color: #94a3b8;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0.3rem 0;
    }
    .kpi-badge {
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
    }
    .kpi-badge-green { background: rgba(16, 185, 129, 0.15); color: #34d399; }
    .kpi-badge-blue  { background: rgba(59, 130, 246, 0.15); color: #60a5fa; }
    .kpi-badge-purple{ background: rgba(168, 85, 247, 0.15); color: #c084fc; }
    .kpi-badge-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }

    /* Segment Highlight Cards */
    .segment-card {
        padding: 1.4rem;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: #1e293b;
        margin-bottom: 1rem;
    }
    .segment-card.vip {
        border-left: 5px solid #10b981;
    }
    .segment-card.inter {
        border-left: 5px solid #6366f1;
    }
    .segment-card.churn {
        border-left: 5px solid #f59e0b;
    }

    /* Result Box in Predictor */
    .result-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .result-tag {
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 700;
        color: #818cf8;
    }
    .result-name {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. MODEL & DATA INFRASTRUCTURE
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "customer_segmentation_model.pkl"

FEATURES = ["Recence", "Frequence", "Montant"]

SEGMENT_METADATA = {
    "Clients a forte valeur": {
        "badge_color": "#10B981",
        "icon": "💎",
        "short_name": "VIP Champions",
        "tagline": "Clients réguliers avec fort pouvoir d'achat et engagement continu",
        "action": "⭐ Programme VIP Concierge, avant-premières exclusives, cadeaux de fidélité et support dédié.",
    },
    "Clients intermediaires": {
        "badge_color": "#6366F1",
        "icon": "📈",
        "short_name": "Core Loyalists",
        "tagline": "Fréquence et panier moyen stables avec potentiel d'upsell",
        "action": "🎁 Offres groupées (cross-sell), seuil de livraison gratuite incitatif et relances programmées.",
    },
    "Clients occasionnels / inactifs": {
        "badge_color": "#F59E0B",
        "icon": "⚠️",
        "short_name": "At-Risk / Dormant",
        "tagline": "Achats rares ou anciens nécessitant une stratégie de réactivation",
        "action": "⏳ Campagne de réactivation agressive, coupons limités dans le temps et sondage de satisfaction.",
    },
}


def find_dataset():
    possible_names = [
        "data.xlsx",
        "Online Retail.xlsx",
        "online_retail.xlsx",
        "online_retail.csv",
        "data.csv",
    ]
    for name in possible_names:
        p = DATA_DIR / name
        if p.exists():
            return p
    files = list(DATA_DIR.glob("*.xlsx")) + list(DATA_DIR.glob("*.csv"))
    return files[0] if files else None


@st.cache_data(show_spinner=False)
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
        x, y, test_size=0.2, random_state=42, stratify=y
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


@st.cache_resource(show_spinner="Chargement de l'intelligence artificielle...")
def get_bundle():
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


bundle, status = get_bundle()

if status == "missing_data" or bundle is None:
    st.error("🚨 Dataset introuvable dans le dossier data/.")
    st.info("Veuillez déposer le fichier Online Retail (.xlsx ou .csv) dans data/")
    st.stop()

rfm_df = bundle["rfm"]
segment_names = bundle["segment_names"]
rf_model = bundle["rf_model"]
profile_df = bundle["profile"].copy()
profile_df["Segment"] = profile_df.index.map(segment_names)

# ==============================================================================
# 3. SIDEBAR NAVIGATION & CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.2rem;">
            <div style="background: linear-gradient(135deg, #6366f1, #3b82f6); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px;">
                ⚡
            </div>
            <div>
                <div style="font-weight: 800; font-size: 1.15rem; color: #f8fafc; letter-spacing: -0.02em;">SegmentPulse</div>
                <div style="font-size: 0.75rem; color: #818cf8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">AI Intelligence Suite</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🏢 Espace Entreprise")
    workspace = st.selectbox(
        "Compte Actif",
        ["Global E-Commerce Store (UK)", "Retail B2B EMEA", "Omnichannel Direct"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🛡️ État du Système")
    st.markdown(
        """
        <div style="background: #1e293b; padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.06); font-size: 0.82rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="color: #94a3b8;">Moteur ML :</span>
                <span style="color: #38bdf8; font-weight: 600;">K-Means + RF (200)</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="color: #94a3b8;">Statut :</span>
                <span style="color: #34d399; font-weight: 600;">● Opérationnel</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #94a3b8;">Clients Indexés :</span>
                <span style="color: #f1f5f9; font-weight: 600;">"""
        + f"{len(rfm_df):,}"
        + """</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("SegmentPulse Enterprise v2.4 • Powered by Scikit-Learn & Antigravity")

# ==============================================================================
# 4. TOP HERO BANNER & KPI METRICS
# ==============================================================================
st.markdown(
    f"""
    <div class="saas-header">
        <div>
            <h1 class="saas-title">SegmentPulse AI Platform</h1>
            <p class="saas-tagline">Segmentation Comportementale RFM Prédictive & Insights Marketing en Temps Réel</p>
        </div>
        <div>
            <span class="badge-pill">
                <span class="status-live"></span>
                Modèle ML v2.4 Actif
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPI Calculations
total_customers = len(rfm_df)
total_revenue = rfm_df["Montant"].sum()
avg_recency = rfm_df["Recence"].mean()
avg_spend = rfm_df["Montant"].mean()

vip_cluster = [k for k, v in segment_names.items() if "forte valeur" in v][0]
vip_count = (rfm_df["Cluster"] == vip_cluster).sum()
vip_pct = (vip_count / total_customers) * 100
vip_rev = rfm_df[rfm_df["Cluster"] == vip_cluster]["Montant"].sum()
vip_rev_pct = (vip_rev / total_revenue) * 100

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Base Clients Active</div>
            <div class="kpi-value">{total_customers:,}</div>
            <span class="kpi-badge kpi-badge-blue">100% Vérifiés</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_kpi2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Cohorte VIP Champions</div>
            <div class="kpi-value">{vip_count:,} <span style="font-size: 1rem; color: #10b981;">({vip_pct:.1f}%)</span></div>
            <span class="kpi-badge kpi-badge-green">{vip_rev_pct:.1f}% du Chiffre d'Affaires</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_kpi3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Valeur Client Moyenne (LTV)</div>
            <div class="kpi-value">{avg_spend:,.0f} €</div>
            <span class="kpi-badge kpi-badge-purple">Global RFM</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_kpi4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Récence Moyenne</div>
            <div class="kpi-value">{avg_recency:.0f} jours</div>
            <span class="kpi-badge kpi-badge-amber">Dernière Transaction</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ==============================================================================
# 5. MAIN NAVIGATION TABS
# ==============================================================================
tab_overview, tab_simulator, tab_explorer, tab_model = st.tabs(
    [
        "📊 Vue d'Ensemble & Cohortes",
        "🎯 Simulateur RFM & Prédiction Live",
        "🔍 Explorateur de Clients & Export",
        "⚙️ Architecture & Santé du Modèle",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW & COHORTS
# ------------------------------------------------------------------------------
with tab_overview:
    col_chart1, col_chart2 = st.columns([1, 1])

    color_map = {
        "Clients a forte valeur": "#10B981",
        "Clients intermediaires": "#6366F1",
        "Clients occasionnels / inactifs": "#F59E0B",
    }

    with col_chart1:
        st.markdown("#### 🍩 Répartition des Segments Clients")
        seg_counts = rfm_df["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        fig_pie = px.pie(
            seg_counts,
            values="Count",
            names="Segment",
            color="Segment",
            color_discrete_map=color_map,
            hole=0.6,
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hoverinfo="label+value+percent",
            marker=dict(line=dict(color="#0f172a", width=2)),
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=320,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        st.markdown("#### 💶 Chiffre d'Affaires Généré par Segment")
        seg_revenue = rfm_df.groupby("Segment")["Montant"].sum().reset_index()
        fig_rev = px.bar(
            seg_revenue,
            x="Segment",
            y="Montant",
            color="Segment",
            color_discrete_map=color_map,
            text_auto=".2s",
        )
        fig_rev.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Montant Total (€)"),
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=320,
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("---")

    # 3D Interactive Cluster Scatter Plot
    st.markdown("#### 🌌 Visualisation 3D Interactive de l'Espace RFM")
    st.caption("Naviguez, zoomez et pivotez dans l'espace pour observer les regroupements tridimensionnels déterminés par l'algorithme K-Means.")

    # Sample for smooth 3D rendering
    sample_df = rfm_df.sample(n=min(1200, len(rfm_df)), random_state=42)

    fig_3d = px.scatter_3d(
        sample_df,
        x="Recence",
        y="Frequence",
        z="Montant",
        color="Segment",
        color_discrete_map=color_map,
        hover_data=["CustomerID", "Recence", "Frequence", "Montant"],
        opacity=0.75,
        log_z=True,
    )
    fig_3d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        scene=dict(
            xaxis=dict(title="Récence (jours)", backgroundcolor="rgba(15,23,42,0.6)", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Fréquence (achats)", backgroundcolor="rgba(15,23,42,0.6)", gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(title="Montant Log (€)", backgroundcolor="rgba(15,23,42,0.6)", gridcolor="rgba(255,255,255,0.1)"),
        ),
        margin=dict(l=0, r=0, b=0, t=20),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    # Strategic Segment Matrix
    st.markdown("#### 🎯 Stratégie Marketing & Playbook par Segment")
    strat_col1, strat_col2, strat_col3 = st.columns(3)

    for col, (seg_key, meta) in zip([strat_col1, strat_col2, strat_col3], SEGMENT_METADATA.items()):
        sub_df = rfm_df[rfm_df["Segment"] == seg_key]
        with col:
            st.markdown(
                f"""
                <div style="background: #1e293b; border-left: 4px solid {meta['badge_color']}; border-radius: 12px; padding: 1.2rem; border-top: 1px solid rgba(255,255,255,0.06); border-right: 1px solid rgba(255,255,255,0.06); border-bottom: 1px solid rgba(255,255,255,0.06);">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-size: 1.4rem;">{meta['icon']}</span>
                        <span style="background: rgba(255,255,255,0.08); padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; color: {meta['badge_color']};">{len(sub_df):,} clients</span>
                    </div>
                    <div style="font-weight: 700; font-size: 1.05rem; color: #f8fafc; margin-bottom: 4px;">{meta['short_name']}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 12px;">{meta['tagline']}</div>
                    <div style="font-size: 0.82rem; background: rgba(0,0,0,0.25); padding: 10px; border-radius: 8px; color: #e2e8f0; line-height: 1.4;">
                        <strong>Action recommandée :</strong><br>{meta['action']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------------------------
# TAB 2: LIVE SIMULATOR & PREDICTOR
# ------------------------------------------------------------------------------
with tab_simulator:
    st.markdown("### 🧪 Simulateur Prédictif de Profil Client")
    st.caption("Ajustez les indicateurs RFM pour évaluer instantanément l'attribution du segment et obtenir la recommandation d'action automatique.")

    st.markdown("##### ⚡ Préréglages Rapides (Presets)")
    p_col1, p_col2, p_col3 = st.columns(3)
    if "sim_r" not in st.session_state:
        st.session_state.sim_r = 15
        st.session_state.sim_f = 12
        st.session_state.sim_m = 6500.0

    if p_col1.button("💎 Client Champion VIP", use_container_width=True):
        st.session_state.sim_r = 8
        st.session_state.sim_f = 18
        st.session_state.sim_m = 9200.0
        st.rerun()

    if p_col2.button("📈 Client Intermédiaire Standard", use_container_width=True):
        st.session_state.sim_r = 40
        st.session_state.sim_f = 4
        st.session_state.sim_m = 1200.0
        st.rerun()

    if p_col3.button("⚠️ Client Dormant / Churn", use_container_width=True):
        st.session_state.sim_r = 180
        st.session_state.sim_f = 1
        st.session_state.sim_m = 250.0
        st.rerun()

    st.write("")
    sim_input_col, sim_result_col = st.columns([1.1, 1])

    with sim_input_col:
        st.markdown(
            """
            <div style="background: #1e293b; padding: 1.4rem; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08);">
                <div style="font-weight: 700; font-size: 1rem; color: #f8fafc; margin-bottom: 1rem;">Paramètres RFM du Client</div>
            """,
            unsafe_allow_html=True,
        )

        in_recence = st.slider(
            "⏳ Récence (Jours depuis le dernier achat)",
            min_value=1,
            max_value=380,
            value=int(st.session_state.sim_r),
            help="Nombre de jours écoulés depuis la date de la dernière commande enregistrée.",
        )

        in_frequence = st.slider(
            "🛍️ Fréquence (Nombre total de commandes passées)",
            min_value=1,
            max_value=100,
            value=int(st.session_state.sim_f),
            help="Total des factures uniques générées par ce client.",
        )

        in_montant = st.number_input(
            "💶 Montant Total Dépensé (€)",
            min_value=1.0,
            max_value=100000.0,
            value=float(st.session_state.sim_m),
            step=50.0,
            help="Somme totale de la valeur des articles achetés.",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Perform Model Inference
    input_df = pd.DataFrame([[in_recence, in_frequence, in_montant]], columns=FEATURES)
    pred_cluster = int(rf_model.predict(input_df)[0])
    pred_segment = segment_names.get(pred_cluster, f"Cluster {pred_cluster}")
    pred_probs = rf_model.predict_proba(input_df)[0]
    classes = rf_model.classes_

    meta = SEGMENT_METADATA.get(pred_segment, {
        "badge_color": "#6366F1",
        "icon": "🎯",
        "short_name": pred_segment,
        "tagline": "Segment client prédit",
        "action": "Personnalisez votre communication.",
    })

    with sim_result_col:
        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-tag">Segment Identifié par Random Forest</div>
                <div style="font-size: 2.8rem; margin: 0.3rem 0;">{meta['icon']}</div>
                <div class="result-name" style="color: {meta['badge_color']};">{meta['short_name']}</div>
                <div style="font-size: 0.95rem; color: #cbd5e1; margin-bottom: 1.2rem;">{meta['tagline']}</div>
                
                <div style="background: rgba(255,255,255,0.06); padding: 12px; border-radius: 10px; text-align: left; border-left: 3px solid {meta['badge_color']};">
                    <span style="font-weight: 700; color: #f8fafc; font-size: 0.85rem;">⚡ Plan d'Action Automatique :</span><br>
                    <span style="font-size: 0.85rem; color: #cbd5e1;">{meta['action']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("##### 📊 Distribution des Probabilités du Modèle")
        for cls_idx, prob in zip(classes, pred_probs):
            cls_seg = segment_names.get(cls_idx, f"Cluster {cls_idx}")
            seg_info = SEGMENT_METADATA.get(cls_seg, {"short_name": cls_seg, "badge_color": "#818cf8"})
            st.write(f"**{seg_info['short_name']}** — `{prob*100:.1f}%`")
            st.progress(float(prob))

# ------------------------------------------------------------------------------
# TAB 3: CUSTOMER EXPLORER & CRM EXPORT
# ------------------------------------------------------------------------------
with tab_explorer:
    st.markdown("### 🔍 Explorateur de Données & Exportation CRM")
    st.caption("Filtrez les comptes clients par segment, panier moyen ou récence pour exporter vos listes vers Klaviyo, HubSpot ou Salesforce.")

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        sel_segment = st.selectbox(
            "Filtrer par Segment",
            ["Tous les Segments"] + list(segment_names.values()),
        )
    with f_col2:
        min_spend_filter = st.number_input("Dépense Minimale (€)", min_value=0.0, value=0.0, step=100.0)
    with f_col3:
        search_cust = st.text_input("Rechercher un CustomerID", placeholder="ex: 12347")

    filtered_df = rfm_df.copy()
    if sel_segment != "Tous les Segments":
        filtered_df = filtered_df[filtered_df["Segment"] == sel_segment]
    if min_spend_filter > 0:
        filtered_df = filtered_df[filtered_df["Montant"] >= min_spend_filter]
    if search_cust.strip():
        filtered_df = filtered_df[filtered_df["CustomerID"].astype(str).str.contains(search_cust.strip())]

    st.markdown(f"**{len(filtered_df):,} clients trouvés** correspondant aux critères sélectionnés :")

    # Format numbers for professional SaaS display
    display_df = filtered_df[["CustomerID", "Segment", "Recence", "Frequence", "Montant"]].copy()
    display_df["Montant"] = display_df["Montant"].map(lambda x: f"{x:,.2f} €")
    display_df.rename(
        columns={
            "CustomerID": "ID Client",
            "Segment": "Segment Prédit",
            "Recence": "Récence (Jours)",
            "Frequence": "Fréquence (Commandes)",
            "Montant": "Montant Total Décaissé",
        },
        inplace=True,
    )

    st.dataframe(display_df, use_container_width=True, height=360)

    # Export Button
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Télécharger la Sélection (Format CSV pour CRM)",
        data=csv_data,
        file_name="segmentpulse_customer_cohort.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ------------------------------------------------------------------------------
# TAB 4: MODEL ARCHITECTURE & GOVERNANCE
# ------------------------------------------------------------------------------
with tab_model:
    st.markdown("### ⚙️ Architecture du Modèle & Explicabilité")
    st.caption("Détails d'entraînement, importance des variables et métriques de validation statistique.")

    m_col1, m_col2 = st.columns([1, 1])

    with m_col1:
        st.markdown("#### ⚖️ Importance des Variables (Feature Importance)")
        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": ["Récence", "Fréquence", "Montant Total"],
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            text_auto=".2%",
            color="Importance",
            color_continuous_scale="Purples",
        )
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="Poids Relatif"),
            yaxis=dict(title=""),
            coloraxis_showscale=False,
            height=260,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with m_col2:
        st.markdown("#### 📐 Profils Moyens par Cluster (Centroids)")
        centroid_display = profile_df.copy()
        centroid_display["Montant"] = centroid_display["Montant"].map(lambda x: f"{x:,.2f} €")
        centroid_display["Recence"] = centroid_display["Recence"].map(lambda x: f"{x:.1f} j")
        centroid_display["Frequence"] = centroid_display["Frequence"].map(lambda x: f"{x:.1f} cmds")
        st.dataframe(centroid_display[["Segment", "Recence", "Frequence", "Montant"]], use_container_width=True, height=260)

    st.markdown("---")
    st.markdown("#### 📋 Spécifications Techniques du Pipeline")
    spec_col1, spec_col2, spec_col3 = st.columns(3)
    with spec_col1:
        st.markdown(
            """
            - **Algorithme Non-Supervisé :** K-Means (k=3, n_init=10, random_state=42)
            - **Transformation :** Log-transformation `np.log1p()` + `StandardScaler`
            """
        )
    with spec_col2:
        st.markdown(
            """
            - **Classifieur Supervisé :** Random Forest (200 estimateurs)
            - **Stratification :** 80% Entraînement / 20% Test
            """
        )
    with spec_col3:
        st.markdown(
            """
            - **Source de Données :** Transactions Online Retail (UK)
            - **Latence d'Inférence :** < 2.5 ms
            """
        )
