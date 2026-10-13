from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & HARMONIOUS SKY-BLUE THEME
# ==============================================================================
st.set_page_config(
    page_title="SegmentPulse | Customer RFM Intelligence",
    page_icon="https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# High-Contrast Harmonious Sky-Blue Palette
# Background: #F0F9FF (Soft Sky) | Cards: #FFFFFF | Text: #0F172A (Deep Slate Navy) | Accents: #0284C7, #0369A1
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global reset for maximum contrast & harmony */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        background-color: #f0f9ff !important;
        color: #0f172a !important;
    }

    /* Force all headings to high-contrast deep slate navy */
    h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown span {
        color: #0f172a !important;
    }

    /* Sidebar full harmonization */
    section[data-testid="stSidebar"] {
        background-color: #e0f2fe !important;
        border-right: 1px solid #bae6fd !important;
    }
    section[data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] a {
        color: #0284c7 !important;
        font-weight: 600 !important;
        text-decoration: underline !important;
    }

    /* Navbar Container */
    .saas-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        border: 2px solid #bae6fd;
        border-radius: 14px;
        padding: 1.4rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.08);
    }
    .brand-title {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: #0369a1 !important;
        letter-spacing: -0.02em;
        margin: 0 !important;
    }
    .brand-subtitle {
        font-size: 0.92rem !important;
        color: #334155 !important;
        margin-top: 4px !important;
        margin-bottom: 0 !important;
        font-weight: 500 !important;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #e0f2fe;
        color: #0369a1 !important;
        border: 1.5px solid #7dd3fc;
        padding: 0.45rem 1rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .dot-pulse {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #0284c7;
        box-shadow: 0 0 6px #0284c7;
    }

    /* KPI Cards */
    .kpi-card {
        background: #ffffff;
        border: 1.5px solid #bae6fd;
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 3px 10px rgba(2, 132, 199, 0.06);
        transition: all 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #0284c7;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(2, 132, 199, 0.12);
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
        color: #475569;
    }
    .kpi-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.3rem 0;
    }
    .kpi-sub {
        font-size: 0.8rem;
        font-weight: 600;
        color: #0284c7;
    }

    /* Content Panels */
    .panel-card {
        background: #ffffff;
        border: 1.5px solid #bae6fd;
        border-radius: 14px;
        padding: 1.4rem;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.05);
        margin-bottom: 1.2rem;
    }
    .panel-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.8rem;
        border-bottom: 1.5px solid #e0f2fe;
        padding-bottom: 0.5rem;
    }

    /* Result Panel in Predictor */
    .result-panel {
        background: #ffffff;
        border: 2px solid #0284c7;
        border-radius: 14px;
        padding: 1.6rem;
        text-align: center;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.1);
    }
    .result-badge {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #0369a1;
        background: #e0f2fe;
        border: 1px solid #bae6fd;
        padding: 4px 12px;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
    .result-title {
        font-size: 2rem;
        font-weight: 800;
        color: #0284c7;
        margin: 0.3rem 0;
    }
    .result-description {
        font-size: 0.9rem;
        color: #334155;
        margin-bottom: 1.1rem;
        font-weight: 500;
    }
    .action-box {
        background: #f0f9ff;
        border: 1.5px solid #bae6fd;
        border-radius: 8px;
        padding: 1rem;
        text-align: left;
        font-size: 0.86rem;
        color: #0f172a;
        line-height: 1.5;
    }

    /* Streamlit Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 2px solid #bae6fd;
        margin-bottom: 1.2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        color: #475569 !important;
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-bottom: none !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 18px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #0284c7 !important;
        background: #e0f2fe !important;
        border: 1.5px solid #0284c7 !important;
        border-bottom: none !important;
    }

    /* Primary buttons */
    .stButton>button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        transition: background-color 0.2s ease !important;
    }
    .stButton>button:hover {
        background-color: #0369a1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. DATA PROCESSING & MODEL PIPELINE
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "customer_segmentation_model.pkl"

FEATURES = ["Recence", "Frequence", "Montant"]

SEGMENT_DETAILS = {
    "Clients a forte valeur": {
        "color": "#0284c7",
        "label": "High-Value Champions",
        "desc": "Frequent buyers with substantial total lifetime expenditure.",
        "strategy": "Dedicated VIP loyalty tier, priority support access, and early product previews.",
    },
    "Clients intermediaires": {
        "color": "#0ea5e9",
        "label": "Core Regular Buyers",
        "desc": "Reliable repeat customers with consistent purchase cycles.",
        "strategy": "Product bundle recommendations, free shipping thresholds, and targeted cross-sell workflows.",
    },
    "Clients occasionnels / inactifs": {
        "color": "#64748b",
        "label": "At-Risk / Dormant Accounts",
        "desc": "High inactivity duration with minimal historical transactions.",
        "strategy": "Automated re-engagement sequences, limited-time return incentives, and feedback surveys.",
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


@st.cache_resource(show_spinner=False)
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
    st.error("Dataset not found in data/ directory.")
    st.info("Place Online Retail (.xlsx or .csv) inside data/ to initialize.")
    st.stop()

rfm_df = bundle["rfm"]
segment_names = bundle["segment_names"]
rf_model = bundle["rf_model"]
profile_df = bundle["profile"].copy()
profile_df["Segment"] = profile_df.index.map(segment_names)

# ==============================================================================
# 3. SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding-bottom: 0.8rem; border-bottom: 1.5px solid #bae6fd; margin-bottom: 1.2rem;">
            <div style="font-size: 1.25rem; font-weight: 800; color: #0369a1; letter-spacing: -0.02em;">SegmentPulse</div>
            <div style="font-size: 0.82rem; color: #334155; font-weight: 600;">Enterprise RFM Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<strong style='color:#0f172a;'>Storefront Scope</strong>", unsafe_allow_html=True)
    selected_org = st.selectbox(
        "Store Scope",
        ["Production Storefront (UK)", "Wholesale Direct", "Retail Partner Network"],
        label_visibility="collapsed",
    )

    st.markdown("<br><strong style='color:#0f172a;'>System Information</strong>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background: #ffffff; border: 1.5px solid #bae6fd; border-radius: 10px; padding: 12px; font-size: 0.82rem; line-height: 1.7; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#475569; font-weight: 500;">Classifier:</span>
                <span style="font-weight:700; color:#0f172a;">Random Forest (200)</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#475569; font-weight: 500;">Clusters:</span>
                <span style="font-weight:700; color:#0f172a;">K-Means (k=3)</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#475569; font-weight: 500;">Indexed Accounts:</span>
                <span style="font-weight:800; color:#0284c7;">{len(rfm_df):,}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br><strong style='color:#0f172a;'>Project References</strong>", unsafe_allow_html=True)
    st.markdown(
        """
        - [GitHub Repository](https://github.com/ilyasaitbel/customer-segmentation)
        - [Local Source (app.py)](file:///C:/Users/Youcode/.gemini/antigravity/scratch/customer-segmentation/app.py)
        - [Model Binary (.pkl)](file:///C:/Users/Youcode/.gemini/antigravity/scratch/customer-segmentation/models/customer_segmentation_model.pkl)
        """
    )
    st.caption("Release 2.4 • Managed Environment")

# ==============================================================================
# 4. TOP NAVBAR & CORE METRIC STRIP
# ==============================================================================
st.markdown(
    """
    <div class="saas-navbar">
        <div>
            <h1 class="brand-title">SegmentPulse Analytics</h1>
            <p class="brand-subtitle">Automated Behavioral Customer Segmentation & RFM Lifecycle Management</p>
        </div>
        <div>
            <div class="status-badge">
                <span class="dot-pulse"></span>
                Production System Active
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Metric computations
total_count = len(rfm_df)
total_gross = rfm_df["Montant"].sum()
avg_recency = rfm_df["Recence"].mean()
avg_revenue = rfm_df["Montant"].mean()

vip_cluster_id = [k for k, v in segment_names.items() if "forte valeur" in v][0]
vip_records = (rfm_df["Cluster"] == vip_cluster_id).sum()
vip_percent = (vip_records / total_count) * 100
vip_gross = rfm_df[rfm_df["Cluster"] == vip_cluster_id]["Montant"].sum()
vip_gross_share = (vip_gross / total_gross) * 100

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

with kpi_c1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Active Customer Accounts</div>
            <div class="kpi-val">{total_count:,}</div>
            <div class="kpi-sub">Indexed in Database</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">High-Value Segment</div>
            <div class="kpi-val">{vip_records:,} <span style="font-size: 0.95rem; color: #0284c7; font-weight: 600;">({vip_percent:.1f}%)</span></div>
            <div class="kpi-sub">{vip_gross_share:.1f}% of total revenue</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Mean Lifetime Value</div>
            <div class="kpi-val">{avg_revenue:,.2f} €</div>
            <div class="kpi-sub">Average per Customer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_c4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Mean Recency Window</div>
            <div class="kpi-val">{avg_recency:.1f} days</div>
            <div class="kpi-sub">Since Last Transaction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ==============================================================================
# 5. TABS WITH HIGH-CONTRAST PALETTE
# ==============================================================================
tab_analytics, tab_sim, tab_data, tab_sys = st.tabs(
    [
        "Cohort Analytics",
        "Customer Simulator",
        "Customer Directory",
        "Model Governance",
    ]
)

sky_colors = {
    "Clients a forte valeur": "#0284c7",       # Vibrant Sky Blue
    "Clients intermediaires": "#38bdf8",       # Light Azure
    "Clients occasionnels / inactifs": "#94a3b8", # Crisp Slate Grey
}

# ------------------------------------------------------------------------------
# TAB 1: COHORT ANALYTICS
# ------------------------------------------------------------------------------
with tab_analytics:
    ch_col1, ch_col2 = st.columns([1, 1])

    with ch_col1:
        st.markdown("<h4 style='color:#0f172a; margin-bottom: 0.5rem;'>Account Distribution by Segment</h4>", unsafe_allow_html=True)
        counts = rfm_df["Segment"].value_counts().reset_index()
        counts.columns = ["Segment", "Count"]

        fig_pie = px.pie(
            counts,
            values="Count",
            names="Segment",
            color="Segment",
            color_discrete_map=sky_colors,
            hole=0.62,
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hoverinfo="label+value+percent",
            marker=dict(line=dict(color="#ffffff", width=2)),
        )
        fig_pie.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a", family="Plus Jakarta Sans", size=13),
            showlegend=False,
            margin=dict(t=15, b=15, l=15, r=15),
            height=320,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with ch_col2:
        st.markdown("<h4 style='color:#0f172a; margin-bottom: 0.5rem;'>Total Gross Revenue by Segment</h4>", unsafe_allow_html=True)
        rev_by_seg = rfm_df.groupby("Segment")["Montant"].sum().reset_index()

        fig_bar = px.bar(
            rev_by_seg,
            x="Segment",
            y="Montant",
            color="Segment",
            color_discrete_map=sky_colors,
            text_auto=".2s",
        )
        fig_bar.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a", family="Plus Jakarta Sans", size=13),
            xaxis=dict(showgrid=False, title="", tickfont=dict(color="#0f172a", size=12)),
            yaxis=dict(showgrid=True, gridcolor="#e2e8f0", title="Gross Revenue (€)", title_font=dict(color="#0f172a")),
            showlegend=False,
            margin=dict(t=15, b=15, l=15, r=15),
            height=320,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("<h4 style='color:#0f172a;'>Interactive 3D RFM Clustering Space</h4>", unsafe_allow_html=True)
    st.caption("Rotate, pan, and zoom to explore how K-Means partitions customers across Recency, Frequency, and Monetary dimensions.")

    sample_rfm = rfm_df.sample(n=min(1200, len(rfm_df)), random_state=42)

    fig_3d = px.scatter_3d(
        sample_rfm,
        x="Recence",
        y="Frequence",
        z="Montant",
        color="Segment",
        color_discrete_map=sky_colors,
        hover_data=["CustomerID", "Recence", "Frequence", "Montant"],
        opacity=0.8,
        log_z=True,
    )
    fig_3d.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Plus Jakarta Sans"),
        scene=dict(
            xaxis=dict(title="Recency (Days)", backgroundcolor="#f0f9ff", gridcolor="#bae6fd", tickfont=dict(color="#0f172a")),
            yaxis=dict(title="Frequency (Orders)", backgroundcolor="#f0f9ff", gridcolor="#bae6fd", tickfont=dict(color="#0f172a")),
            zaxis=dict(title="Monetary Log (€)", backgroundcolor="#f0f9ff", gridcolor="#bae6fd", tickfont=dict(color="#0f172a")),
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#0f172a")),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("<h4 style='color:#0f172a;'>Lifecycle Marketing Strategic Playbooks</h4>", unsafe_allow_html=True)
    strat1, strat2, strat3 = st.columns(3)

    for col, (k, meta) in zip([strat1, strat2, strat3], SEGMENT_DETAILS.items()):
        sub_count = (rfm_df["Segment"] == k).sum()
        with col:
            st.markdown(
                f"""
                <div class="panel-card" style="border-top: 4px solid {meta['color']};">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
                        <span style="font-weight:800; color:#0f172a; font-size: 1.05rem;">{meta['label']}</span>
                        <span style="font-size:0.75rem; background:#e0f2fe; color:#0369a1; border: 1px solid #bae6fd; padding:3px 8px; border-radius:6px; font-weight:700;">{sub_count:,} accounts</span>
                    </div>
                    <div style="font-size:0.84rem; color:#475569; margin-bottom:12px; font-weight: 500;">{meta['desc']}</div>
                    <div style="background:#f0f9ff; border: 1.5px solid #bae6fd; border-radius:8px; padding:12px; font-size:0.84rem; color:#0f172a; line-height:1.5;">
                        <strong style="color:#0284c7;">Action Framework:</strong><br>{meta['strategy']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------------------------
# TAB 2: CUSTOMER SIMULATOR
# ------------------------------------------------------------------------------
with tab_sim:
    st.markdown("<h4 style='color:#0f172a;'>Real-Time Behavioral RFM Predictor</h4>", unsafe_allow_html=True)
    st.caption("Adjust customer behavioral indicators or select a preset profile to test the classification engine.")

    b_col1, b_col2, b_col3 = st.columns(3)
    if "s_r" not in st.session_state:
        st.session_state.s_r = 14
        st.session_state.s_f = 12
        st.session_state.s_m = 6800.0

    if b_col1.button("Preset: High-Value VIP Client", use_container_width=True):
        st.session_state.s_r = 8
        st.session_state.s_f = 16
        st.session_state.s_m = 8900.0
        st.rerun()

    if b_col2.button("Preset: Regular Core Buyer", use_container_width=True):
        st.session_state.s_r = 38
        st.session_state.s_f = 4
        st.session_state.s_m = 1350.0
        st.rerun()

    if b_col3.button("Preset: At-Risk Inactive Account", use_container_width=True):
        st.session_state.s_r = 195
        st.session_state.s_f = 1
        st.session_state.s_m = 280.0
        st.rerun()

    st.write("")
    in_col, out_col = st.columns([1.1, 1])

    with in_col:
        st.markdown(
            """
            <div class="panel-card">
                <div class="panel-title">Behavioral Input Parameters</div>
            """,
            unsafe_allow_html=True,
        )

        val_recence = st.slider(
            "Recency Window (Days elapsed since last purchase)",
            min_value=1,
            max_value=380,
            value=int(st.session_state.s_r),
            help="Days since the customer's most recent completed order.",
        )

        val_frequence = st.slider(
            "Order Frequency (Lifetime completed invoices)",
            min_value=1,
            max_value=100,
            value=int(st.session_state.s_f),
            help="Total count of unique historical transactions.",
        )

        val_montant = st.number_input(
            "Cumulative Expenditure (€)",
            min_value=1.0,
            max_value=100000.0,
            value=float(st.session_state.s_m),
            step=50.0,
            help="Total gross currency spent across all historical invoices.",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Inference
    sample_input = pd.DataFrame([[val_recence, val_frequence, val_montant]], columns=FEATURES)
    cluster_pred = int(rf_model.predict(sample_input)[0])
    segment_pred = segment_names.get(cluster_pred, f"Cluster {cluster_pred}")
    probs = rf_model.predict_proba(sample_input)[0]
    class_list = rf_model.classes_

    details = SEGMENT_DETAILS.get(segment_pred, {
        "color": "#0284c7",
        "label": segment_pred,
        "desc": "Predicted customer segment.",
        "strategy": "Engage according to organizational guidelines.",
    })

    with out_col:
        st.markdown(
            f"""
            <div class="result-panel">
                <span class="result-badge">Random Forest Output</span>
                <div class="result-title">{details['label']}</div>
                <div class="result-description">{details['desc']}</div>
                <div class="action-box">
                    <strong style="color: #0369a1;">Automated Next-Best-Action:</strong><br>
                    {details['strategy']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("<h5 style='color:#0f172a;'>Class Probability Distribution</h5>", unsafe_allow_html=True)
        for c_idx, p_val in zip(class_list, probs):
            c_name = segment_names.get(c_idx, f"Cluster {c_idx}")
            c_label = SEGMENT_DETAILS.get(c_name, {}).get("label", c_name)
            st.markdown(f"<strong style='color:#0f172a;'>{c_label}</strong> — <code style='color:#0369a1; font-weight:700;'>{p_val*100:.1f}%</code>", unsafe_allow_html=True)
            st.progress(float(p_val))

# ------------------------------------------------------------------------------
# TAB 3: CUSTOMER DIRECTORY & EXPORT
# ------------------------------------------------------------------------------
with tab_data:
    st.markdown("<h4 style='color:#0f172a;'>Customer Database Directory</h4>", unsafe_allow_html=True)
    st.caption("Filter and export segmented customer accounts for synchronisation with marketing automation and CRM tooling.")

    f1, f2, f3 = st.columns(3)
    with f1:
        filter_segment = st.selectbox(
            "Cohort Filter",
            ["All Cohorts"] + list(segment_names.values()),
        )
    with f2:
        filter_min_spend = st.number_input("Minimum Lifetime Spend (€)", min_value=0.0, value=0.0, step=100.0)
    with f3:
        search_id = st.text_input("Lookup Account (CustomerID)", placeholder="e.g. 12347")

    data_view = rfm_df.copy()
    if filter_segment != "All Cohorts":
        data_view = data_view[data_view["Segment"] == filter_segment]
    if filter_min_spend > 0:
        data_view = data_view[data_view["Montant"] >= filter_min_spend]
    if search_id.strip():
        data_view = data_view[data_view["CustomerID"].astype(str).str.contains(search_id.strip())]

    st.markdown(f"<strong style='color:#0f172a;'>{len(data_view):,} accounts match the active filter criteria</strong>", unsafe_allow_html=True)

    formatted_table = data_view[["CustomerID", "Segment", "Recence", "Frequence", "Montant"]].copy()
    formatted_table["Montant"] = formatted_table["Montant"].map(lambda x: f"{x:,.2f} €")
    formatted_table.rename(
        columns={
            "CustomerID": "Customer ID",
            "Segment": "Segment Classification",
            "Recence": "Recency (Days)",
            "Frequence": "Order Count",
            "Montant": "Gross Total (€)",
        },
        inplace=True,
    )

    st.dataframe(formatted_table, use_container_width=True, height=360)

    csv_payload = data_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export Active Cohort (CSV)",
        data=csv_payload,
        file_name="segmentpulse_customer_export.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ------------------------------------------------------------------------------
# TAB 4: MODEL GOVERNANCE & DIAGNOSTICS
# ------------------------------------------------------------------------------
with tab_sys:
    st.markdown("<h4 style='color:#0f172a;'>Machine Learning Governance & Explainability</h4>", unsafe_allow_html=True)
    st.caption("Feature contribution factors, cluster centers, and validation metrics.")

    gov_c1, gov_c2 = st.columns([1, 1])

    with gov_c1:
        st.markdown("<h5 style='color:#0f172a;'>Feature Importance (Random Forest Gini Impurity)</h5>", unsafe_allow_html=True)
        feat_imp = rf_model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": ["Recency Window", "Order Frequency", "Monetary Expenditure"],
            "Weight": feat_imp,
        }).sort_values("Weight", ascending=True)

        fig_fi = px.bar(
            imp_df,
            x="Weight",
            y="Feature",
            orientation="h",
            text_auto=".2%",
            color="Weight",
            color_continuous_scale="Blues",
        )
        fig_fi.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a", family="Plus Jakarta Sans", size=13),
            xaxis=dict(showgrid=True, gridcolor="#e2e8f0", title="Relative Weight", title_font=dict(color="#0f172a")),
            yaxis=dict(title="", tickfont=dict(color="#0f172a", size=12)),
            coloraxis_showscale=False,
            height=260,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with gov_c2:
        st.markdown("<h5 style='color:#0f172a;'>Cluster Centroids (Mean Baseline)</h5>", unsafe_allow_html=True)
        centroid_view = profile_df.copy()
        centroid_view["Montant"] = centroid_view["Montant"].map(lambda x: f"{x:,.2f} €")
        centroid_view["Recence"] = centroid_view["Recence"].map(lambda x: f"{x:.1f} days")
        centroid_view["Frequence"] = centroid_view["Frequence"].map(lambda x: f"{x:.1f} orders")
        centroid_view.rename(
            columns={
                "Segment": "Segment",
                "Recence": "Mean Recency",
                "Frequence": "Mean Frequency",
                "Montant": "Mean Spend",
            },
            inplace=True,
        )
        st.dataframe(centroid_view[["Segment", "Mean Recency", "Mean Frequency", "Mean Spend"]], use_container_width=True, height=260)

    st.markdown("---")
    st.markdown("<h5 style='color:#0f172a;'>Architecture References & Governance</h5>", unsafe_allow_html=True)
    st.markdown(
        """
        - **Pipeline Pipeline:** `Log1p -> StandardScaler -> KMeans(k=3) -> RandomForestClassifier(n_estimators=200)`
        - **Data Integrity:** Deduplicated transaction IDs, cleaned zero/negative pricing, customer ID validation.
        - **Repository Link:** [github.com/ilyasaitbel/customer-segmentation](https://github.com/ilyasaitbel/customer-segmentation)
        """
    )
