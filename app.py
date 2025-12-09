import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, count, sum as spark_sum, max as spark_max, min as spark_min
)
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------
# 1️⃣ Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="🩺 CovInsight Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------
# 2️⃣ Initialize Spark Session
# ----------------------------------------
@st.cache_resource
def get_spark_session():
    """Initialize Spark session with error handling"""
    try:
        spark = (
            SparkSession.builder
            .appName("CovInsight Dashboard")
            .config("spark.ui.port", "4040")
            .config("spark.driver.memory", "1g")
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.sql.adaptive.enabled", "true")
            .getOrCreate()
        )
        return spark
    except Exception as e:
        st.error(f"❌ Failed to initialize Spark: {e}")
        st.info("**Fix:** Ensure Java is installed. Download from https://www.java.com")
        return None

spark = get_spark_session()

if spark is None:
    st.stop()

# ----------------------------------------
# 3️⃣ Header Section
# ----------------------------------------
st.title("🦠 COVID-19 Patient Insights Dashboard")
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("**Analyze cases, vaccination impact, side effects & patient distribution — powered by PySpark + Streamlit**")

# ----------------------------------------
# 4️⃣ Load Dataset with Error Handling
# ----------------------------------------
@st.cache_resource
def load_data():
    """Load and validate COVID dataset"""
    try:
        dataset_path = "covid_dataset.json"
        
        if not os.path.exists(dataset_path):
            st.error(f"❌ Dataset not found: {dataset_path}")
            st.info(f"📁 Current directory: {os.getcwd()}")
            st.warning("**Fix:** Place `covid_dataset.json` in: `c:\\Users\\saran\\Downloads\\spark_1\\`")
            return None
        
        df = spark.read.json(dataset_path)
        
        if df.count() == 0:
            st.error("❌ Dataset is empty")
            return None
        
        expected_cols = {
            "patient_id", "name", "age", "state",
            "new_case", "recovered", "death", "active_case",
            "vaccinated", "vaccine_type", "side_effect", "side_effect_type"
        }
        
        missing = expected_cols - set(df.columns)
        if missing:
            st.warning(f"⚠️ Missing columns: {', '.join(missing)}")
        
        return df
    except Exception as e:
        st.error(f"❌ Could not load dataset: {e}")
        st.info("**Troubleshooting:**\n- Ensure `covid_dataset.json` exists\n- Check file permissions\n- Verify JSON format is valid")
        return None

df = load_data()

if df is None:
    st.stop()

# ----------------------------------------
# 5️⃣ Sidebar Filters
# ----------------------------------------
st.sidebar.header("🔍 Filters")

# Convert to Pandas for filtering
pdf = df.toPandas()

# Handle potential NaN values
pdf["vaccinated"] = pdf["vaccinated"].fillna("Unknown")
pdf["state"] = pdf["state"].fillna("Unknown")
pdf["age"] = pd.to_numeric(pdf["age"], errors='coerce').fillna(0)

# State filter
states = ["All"] + sorted(pdf["state"].unique().tolist())
selected_state = st.sidebar.selectbox("Select State", states)

# Vaccination filter
vac_options = pdf["vaccinated"].unique().tolist()
vac_status = st.sidebar.multiselect(
    "Vaccination Status",
    options=vac_options,
    default=vac_options
)

# Age range filter
age_min = int(pdf["age"].min())
age_max = int(pdf["age"].max())
age_range = st.sidebar.slider(
    "Age Range",
    age_min,
    age_max,
    (age_min, age_max)
)

# Apply filters
filtered_pdf = pdf.copy()
if selected_state != "All":
    filtered_pdf = filtered_pdf[filtered_pdf["state"] == selected_state]

filtered_pdf = filtered_pdf[
    (filtered_pdf["vaccinated"].isin(vac_status)) &
    (filtered_pdf["age"].between(age_range[0], age_range[1]))
]

# ----------------------------------------
# 6️⃣ Patient Risk Segmentation Function
# ----------------------------------------
def classify_risk(row):
    """Classify patient risk level"""
    try:
        if int(row["death"]) == 1:
            return "High Risk (Deceased)"
        if int(row["active_case"]) == 1 and row["vaccinated"] == "No":
            return "High Risk (Unvaccinated)"
        if int(row["active_case"]) == 1 and row["vaccinated"] == "Yes":
            return "Moderate Active Case"
        if int(row["recovered"]) == 1:
            return "Recovered"
        return "Stable / No Case"
    except:
        return "Unknown"

filtered_pdf["RiskSegment"] = filtered_pdf.apply(classify_risk, axis=1)

# ----------------------------------------
# 7️⃣ Overview Metrics Cards
# ----------------------------------------
st.header("📊 COVID Overview Metrics")

col1, col2, col3, col4, col5, col6 = st.columns(6)

total_patients = len(filtered_pdf)
active = int(filtered_pdf["active_case"].sum())
recovered = int(filtered_pdf["recovered"].sum())
deaths = int(filtered_pdf["death"].sum())
vaccinated = (filtered_pdf["vaccinated"] == "Yes").sum()
unvaccinated = (filtered_pdf["vaccinated"] == "No").sum()

with col1:
    st.metric("Total Patients", f"{total_patients:,}")

with col2:
    st.metric("Active Cases", f"{active:,}", delta_color="inverse")

with col3:
    recovery_rate = (recovered / total_patients * 100) if total_patients > 0 else 0
    st.metric("Recovered", f"{recovered:,}", f"{recovery_rate:.1f}%")

with col4:
    mortality_rate = (deaths / total_patients * 100) if total_patients > 0 else 0
    st.metric("Deaths", f"{deaths:,}", f"{mortality_rate:.1f}%", delta_color="inverse")

with col5:
    vac_rate = (vaccinated / total_patients * 100) if total_patients > 0 else 0
    st.metric("Vaccinated", f"{vaccinated:,}", f"{vac_rate:.1f}%")

with col6:
    unvac_rate = (unvaccinated / total_patients * 100) if total_patients > 0 else 0
    st.metric("Unvaccinated", f"{unvaccinated:,}", f"{unvac_rate:.1f}%", delta_color="inverse")

st.divider()

# ----------------------------------------
# 8️⃣ Risk Segmentation & Key Insights
# ----------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🧩 Patient Risk Segmentation")
    
    risk_counts = filtered_pdf["RiskSegment"].value_counts().reset_index()
    risk_counts.columns = ["RiskSegment", "Count"]
    
    seg_colors = {
        "High Risk (Deceased)": "#e74c3c",
        "High Risk (Unvaccinated)": "#e67e22",
        "Moderate Active Case": "#f39c12",
        "Recovered": "#2ecc71",
        "Stable / No Case": "#3498db",
    }
    
    pie_fig = px.pie(
        risk_counts,
        names="RiskSegment",
        values="Count",
        color="RiskSegment",
        color_discrete_map=seg_colors,
        hole=0.4
    )
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(pie_fig, use_container_width=True)

with col_right:
    st.subheader("💉 Vaccination Impact")
    
    vac_df = filtered_pdf.groupby("vaccinated").agg({
        "active_case": "sum",
        "recovered": "sum",
        "death": "sum"
    }).reset_index()
    vac_df.columns = ["Vaccination Status", "Active", "Recovered", "Deaths"]
    
    vac_fig = go.Figure()
    
    vac_fig.add_trace(go.Bar(
        name="Active",
        x=vac_df["Vaccination Status"],
        y=vac_df["Active"],
        marker_color="#f39c12"
    ))
    
    vac_fig.add_trace(go.Bar(
        name="Recovered",
        x=vac_df["Vaccination Status"],
        y=vac_df["Recovered"],
        marker_color="#2ecc71"
    ))
    
    vac_fig.add_trace(go.Bar(
        name="Deaths",
        x=vac_df["Vaccination Status"],
        y=vac_df["Deaths"],
        marker_color="#e74c3c"
    ))
    
    vac_fig.update_layout(barmode="group", height=400)
    st.plotly_chart(vac_fig, use_container_width=True)

st.divider()

# ----------------------------------------
# 9️⃣ State-Wise Analysis - AREA CHART
# ----------------------------------------
st.header("🌍 Geographic Distribution")

if len(filtered_pdf) > 0:
    state_df = filtered_pdf.groupby("state").agg({
        "new_case": "sum",
        "active_case": "sum",
        "recovered": "sum",
        "death": "sum",
        "patient_id": "count"
    }).reset_index()

    state_df.columns = ["State", "New Cases", "Active Cases", "Recovered", "Deaths", "Total Patients"]
    state_df = state_df.sort_values("Total Patients", ascending=False)

    # Create an area chart for state-wise distribution
    state_fig = px.area(
        state_df,
        x="State",
        y=["Active Cases", "Recovered", "Deaths"],
        title="State-wise Case Distribution (Area Chart)",
        color_discrete_sequence=["#FF6B6B", "#4ECDC4", "#45B7D1"]
    )
    state_fig.update_traces(mode='lines+markers')
    state_fig.update_layout(
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(state_fig, use_container_width=True)

    # State-wise table - SIMPLIFIED (no styling)
    st.subheader("📋 State-wise Summary Table")
    st.dataframe(
        state_df.astype(str),
        use_container_width=True
    )
else:
    st.warning("⚠️ No data available for selected filters")

st.divider()

# ----------------------------------------
# 🔟 Age & Demographics Analysis
# ----------------------------------------
st.header("👥 Age & Demographics")

if len(filtered_pdf) > 0:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Age Distribution by Risk Segment")
        
        age_fig = px.histogram(
            filtered_pdf,
            x="age",
            color="RiskSegment",
            nbins=20,
            color_discrete_map=seg_colors,
            barmode="overlay",
            opacity=0.7
        )
        age_fig.update_layout(height=400)
        st.plotly_chart(age_fig, use_container_width=True)

    with col2:
        st.subheader("🧪 Side Effects by Age Group")
        
        filtered_pdf["Age Group"] = pd.cut(
            filtered_pdf["age"],
            bins=[0, 18, 30, 45, 60, 150],
            labels=["0-18", "19-30", "31-45", "46-60", "60+"]
        )
        
        side_age = filtered_pdf.groupby(["Age Group", "side_effect"]).size().reset_index(name="Count")
        
        if len(side_age) > 0:
            side_age_fig = px.sunburst(
                side_age,
                path=["Age Group", "side_effect"],
                values="Count",
                color="Count",
                color_continuous_scale="RdYlGn_r",
                title="Interactive Age Group & Side Effects"
            )
            side_age_fig.update_layout(height=400)
            st.plotly_chart(side_age_fig, use_container_width=True)

st.divider()

# ----------------------------------------
# 1️⃣1️⃣ Vaccine Analysis - TREEMAP & POLAR CHART
# ----------------------------------------
st.header("💉 Vaccine Type Insights")

if len(filtered_pdf) > 0:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Most Used Vaccine Types")
        
        vax_counts = filtered_pdf["vaccine_type"].value_counts().reset_index()
        vax_counts.columns = ["Vaccine Type", "Count"]
        
        if len(vax_counts) > 0:
            vax_fig = px.treemap(
                vax_counts,
                path=["Vaccine Type"],
                values="Count",
                color="Count",
                color_continuous_scale="Tealgrn",
                title="Vaccine Distribution Treemap"
            )
            vax_fig.update_traces(textinfo="label+value+percent root")
            st.plotly_chart(vax_fig, use_container_width=True)

    with col2:
        st.subheader("Side Effect Types")
        
        side_counts = filtered_pdf["side_effect_type"].value_counts().reset_index()
        side_counts.columns = ["Side Effect Type", "Count"]
        
        if len(side_counts) > 0:
            side_fig = px.bar_polar(
                side_counts,
                r="Count",
                theta="Side Effect Type",
                color="Count",
                color_continuous_scale="Plasma",
                title="Side Effect Distribution (Polar)"
            )
            st.plotly_chart(side_fig, use_container_width=True)

# ----------------------------------------
# 1️⃣2️⃣ Detailed Data View
# ----------------------------------------
st.header("📄 Patient Data Explorer")

with st.expander("🔍 View Detailed Patient Records"):
    display_cols = st.multiselect(
        "Select columns to display",
        options=filtered_pdf.columns.tolist(),
        default=["patient_id", "name", "age", "state", "vaccinated", "RiskSegment"]
    )
    
    search = st.text_input("Search by name or patient ID")
    
    display_df = filtered_pdf[display_cols] if display_cols else filtered_pdf
    
    if search:
        display_df = display_df[
            display_df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
        ]
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name="covid_data_filtered.csv",
        mime="text/csv"
    )

# ----------------------------------------
# 1️⃣3️⃣ Footer & Controls
# ----------------------------------------
st.divider()

col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    st.info("💡 **Tip:** Use the sidebar filters to analyze specific segments")

with col2:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

with col3:
    if st.button("🚀 Open Spark UI", use_container_width=True):
        webbrowser.open("http://localhost:4040")
        st.success("✅ Opening Spark UI...")

with col4:
    if st.button("⏹️ Stop Spark", use_container_width=True):
        if spark:
            spark.stop()
            st.success("✅ Spark stopped")

if spark:
    st.caption("🔗 **Spark Web UI:** [http://localhost:4040](http://localhost:4040)")
    
    with st.expander("⚙️ Spark Configuration"):
        try:
            spark_conf = spark.sparkContext.getConf().getAll()
            conf_df = pd.DataFrame(spark_conf, columns=["Property", "Value"])
            st.dataframe(conf_df, use_container_width=True)
            st.markdown(f"**Spark Version:** {spark.version} | **Master:** {spark.sparkContext.master}")
        except Exception as e:
            st.warning(f"Could not load Spark config: {e}")