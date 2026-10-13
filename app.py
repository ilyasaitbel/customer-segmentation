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
# 1. PAGE CONFIGURATION & LIGHT SKY-BLUE SAAS THEME
# ==============================================================================
st.set_page_config(
    page_title="SegmentPulse | Customer RFM Intelligence",
    page_icon="https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Human-Designed Light Sky Blue Theme (Stripe / Tailwind UI style)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0f172a;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #f0f9ff 0%, #f8fafc 400px, #ffffff 100%);
    }

    /* Header Bar */
    .saas-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 1.2rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px -3px rgba(14, 165, 233, 0.08);
    }
    .brand-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0369a1;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 0.88rem;
        color: #64748b;
        margin-top: 2px;
        margin-bottom: 0;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #e0f2fe;
        color: #0284c7;
        border: 1px solid #bae6fd;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .dot-pulse {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #0284c7;
    }

    /* KPI Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e0f2fe;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 2px 6px rgba(14, 165, 233, 0.05);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: #7dd3fc;
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        color: #64748b;
    }
    .kpi-val {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.25rem 0;
    }
    .kpi-sub {
        font-size: 0.78rem;
        font-weight: 500;
        color: #0284c7;
    }

    /* Card Panels */
    .panel-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.4rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 1.2rem;
    }
    .panel-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 0.6rem;
    }

    /* Result Panel in Predictor */
    .result-panel {
        background: #f0f9ff;
        border: 1.5px solid #7dd3fc;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    .result-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #0369a1;
        background: #e0f2fe;
        padding: 3px 10px;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
    .result-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0284c7;
        margin: 0.2rem 0;
    }
    .result-description {
        font-size: 0.88rem;
        color: #475569;
        margin-bottom: 1rem;
    }
    .action-box {
        background: #ffffff;
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 0.9rem;
        text-align: left;
        font-size: 0.84rem;
        color: #1e293b;
        line-height: 1.5;
    }

    /* Custom Streamlit adjustments for light theme */
    div[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        color: #64748b;
        padding: 8px 16px;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #0284c7 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #0284c7 !important;
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
        "color": "#38bdf8",
        "label": "Core Regular Buyers",
        "desc": "Reliable repeat customers with consistent purchase cycles.",
        "strategy": "Product bundle recommendations, free shipping thresholds, and targeted cross-sell workflows.",
    },
    "Clients occasionnels / inactifs": {
        "color": "#94a3b8",
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
# 3. SIDEBAR NAVIGATION & SYSTEM SETTINGS
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding-bottom: 0.8rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.2rem;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #0284c7; letter-spacing: -0.02em;">SegmentPulse</div>
            <div style="font-size: 0.78rem; color: #64748b;">Enterprise RFM Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Workspace Context**")
    selected_org = st.selectbox(
        "Store Scope",
        ["Production Storefront (UK)", "Wholesale Direct", "Retail Partner Network"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**System Information**")
    st.markdown(
        f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 12px; font-size: 0.82rem; line-height: 1.6;">
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748b;">Classifier:</span>
                <span style="font-weight:600; color:#0f172a;">Random Forest (200)</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748b;">Clusters:</span>
                <span style="font-weight:600; color:#0f172a;">K-Means (k=3)</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#64748b;">Records:</span>
                <span style="font-weight:600; color:#0284c7;">{len(rfm_df):,} accounts</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**Documentation & Links**")
    st.markdown(
        """
        - [GitHub Repository](https://github.com/ilyasaitbel/customer-segmentation)
        - [Model Artifacts (PKL)](file:///C:/Users/Youcode/.gemini/antigravity/scratch/customer-segmentation/models/customer_segmentation_model.pkl)
        - [Source Code (app.py)](file:///C:/Users/Youcode/.gemini/antigravity/scratch/customer-segmentation/app.py)
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
            <div class="kpi-val">{vip_records:,} <span style="font-size: 1rem; color: #0284c7; font-weight: 500;">({vip_percent:.1f}%)</span></div>
            <div class="kpi-sub">{vip_gross_share:.1f}% of total gross revenue</div>
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
# 5. WORKSPACE TABS (CLEAN HUMAN-CRAFTED DESIGN)
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
    "Clients a forte valeur": "#0284c7",
    "Clients intermediaires": "#38bdf8",
    "Clients occasionnels / inactifs": "#94a3b8",
}

# ------------------------------------------------------------------------------
# TAB 1: COHORT ANALYTICS
# ------------------------------------------------------------------------------
with tab_analytics:
    ch_col1, ch_col2 = st.columns([1, 1])

    with ch_col1:
        st.markdown("##### Account Distribution by Segment")
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
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", family="Inter"),
            showlegend=False,
            margin=dict(t=15, b=15, l=15, r=15),
            height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with ch_col2:
        st.markdown("##### Total Gross Revenue by Segment")
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
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", family="Inter"),
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Gross Revenue (€)"),
            showlegend=False,
            margin=dict(t=15, b=15, l=15, r=15),
            height=300,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Interactive 3D RFM Clustering Space")
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
        opacity=0.75,
        log_z=True,
    )
    fig_3d.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        font=dict(color="#334155", family="Inter"),
        scene=dict(
            xaxis=dict(title="Recency (Days)", backgroundcolor="#f0f9ff", gridcolor="#e2e8f0"),
            yaxis=dict(title="Frequency (Orders)", backgroundcolor="#f0f9ff", gridcolor="#e2e8f0"),
            zaxis=dict(title="Monetary Log (€)", backgroundcolor="#f0f9ff", gridcolor="#e2e8f0"),
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("##### Lifecycle Marketing Strategic Playbooks")
    strat1, strat2, strat3 = st.columns(3)

    for col, (k, meta) in zip([strat1, strat2, strat3], SEGMENT_DETAILS.items()):
        sub_count = (rfm_df["Segment"] == k).sum()
        with col:
            st.markdown(
                f"""
                <div class="panel-card" style="border-top: 3px solid {meta['color']};">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
                        <span style="font-weight:700; color:#0f172a; font-size: 1rem;">{meta['label']}</span>
                        <span style="font-size:0.75rem; background:#f0f9ff; color:#0284c7; padding:2px 8px; border-radius:4px; font-weight:600;">{sub_count:,} accounts</span>
                    </div>
                    <div style="font-size:0.82rem; color:#64748b; margin-bottom:12px;">{meta['desc']}</div>
                    <div style="background:#f8fafc; border: 1px solid #e2e8f0; border-radius:6px; padding:10px; font-size:0.8rem; color:#1e293b; line-height:1.45;">
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
    st.markdown("##### Real-Time Behavioral RFM Predictor")
    st.caption("Adjust customer behavioral indicators or select a preset profile to test the classification engine.")

    # Presets with clean text labels (no childish emojis)
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
                <div class="panel-header">Behavioral Input Parameters</div>
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
                    <strong style="color: #0284c7;">Automated Next-Best-Action:</strong><br>
                    {details['strategy']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("##### Class Probability Distribution")
        for c_idx, p_val in zip(class_list, probs):
            c_name = segment_names.get(c_idx, f"Cluster {c_idx}")
            c_label = SEGMENT_DETAILS.get(c_name, {}).get("label", c_name)
            st.write(f"**{c_label}** — `{p_val*100:.1f}%`")
            st.progress(float(p_val))

# ------------------------------------------------------------------------------
# TAB 3: CUSTOMER DIRECTORY & EXPORT
# ------------------------------------------------------------------------------
with tab_data:
    st.markdown("##### Customer Database Directory")
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

    st.markdown(f"**{len(data_view):,} accounts match the active filter criteria**")

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
    st.markdown("##### Machine Learning Governance & Explainability")
    st.caption("Feature contribution factors, cluster centers, and validation metrics.")

    gov_c1, gov_c2 = st.columns([1, 1])

    with gov_c1:
        st.markdown("###### Feature Importance (Random Forest Gini Impurity)")
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
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", family="Inter"),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Relative Weight"),
            yaxis=dict(title=""),
            coloraxis_showscale=False,
            height=260,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with gov_c2:
        st.markdown("###### Cluster Centroids (Mean Baseline)")
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
    st.markdown("###### Architecture References & Governance")
    st.markdown(
        """
        - **Pipeline Pipeline:** `Log1p -> StandardScaler -> KMeans(k=3) -> RandomForestClassifier(n_estimators=200)`
        - **Data Integrity:** Deduplicated transaction IDs, cleaned zero/negative pricing, customer ID validation.
        - **Repository Link:** [github.com/ilyasaitbel/customer-segmentation](https://github.com/ilyasaitbel/customer-segmentation)
        """
    )
