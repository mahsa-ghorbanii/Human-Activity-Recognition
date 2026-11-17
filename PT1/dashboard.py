# ============================================
# dashboard.py - Interactive Dashboard for HAR (Section 2-5)
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Page Configuration
st.set_page_config(page_title="HAR Dashboard", page_icon="🏃", layout="wide")

# 2. Data Loading (Cached for performance)
@st.cache_data
def load_data():
    DATA_PATH = 'WISDM_ar_v1.1_raw.txt'
    records = []
    with open(DATA_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            for seg in line.split(';'):
                seg = seg.strip().strip(',')
                if not seg: continue
                parts = [p.strip() for p in seg.split(',') if p.strip() != '']
                if len(parts) == 6:
                    records.append(parts)
    
    df = pd.DataFrame(records, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])
    df['user'] = df['user'].astype(int)
    for col in ['timestamp', 'x', 'y', 'z']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna().sort_values('timestamp').reset_index(drop=True)
    return df

# Load data
with st.spinner("Loading data..."):
    df = load_data()

# 3. Sidebar - User Interface
st.sidebar.header("⚙️ Settings")
st.sidebar.markdown("Select parameters to compare signals.")

activities = sorted(df['activity'].unique())
users = sorted(df['user'].unique())

st.sidebar.subheader("Top Plot")
act1 = st.sidebar.selectbox("Activity 1", activities, index=activities.index('Walking') if 'Walking' in activities else 0)
user1 = st.sidebar.selectbox("User 1", users, index=users.index(3) if 3 in users else 0)

st.sidebar.subheader("Bottom Plot (Compare)")
act2 = st.sidebar.selectbox("Activity 2", activities, index=activities.index('Jogging') if 'Jogging' in activities else 1)
user2 = st.sidebar.selectbox("User 2", users, index=users.index(3) if 3 in users else 0)

num_samples = st.sidebar.slider("Number of samples (20Hz)", min_value=51, max_value=1000, value=200, step=50)

# 4. Main Page
st.title("🏃 Human Activity Recognition Dashboard")
st.markdown("Compare accelerometer signals across different activities and users.")

# Function to get filtered data
def get_data(act, usr):
    subset = df[(df['activity'] == act) & (df['user'] == usr)]
    return subset.iloc[:num_samples]

sub1 = get_data(act1, user1)
sub2 = get_data(act2, user2)

# Create subplots
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=(f"{act1} - User {user1}", f"{act2} - User {user2}"),
    shared_xaxes=True,
    vertical_spacing=0.1
)

# Add traces for Plot 1
if not sub1.empty:
    t1 = np.arange(len(sub1)) / 20.0
    fig.add_trace(go.Scatter(x=t1, y=sub1['x'], name='X', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t1, y=sub1['y'], name='Y', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t1, y=sub1['z'], name='Z', line=dict(color='green')), row=1, col=1)
else:
    fig.add_annotation(text="No data for this user/activity", row=1, col=1)

# Add traces for Plot 2
if not sub2.empty:
    t2 = np.arange(len(sub2)) / 20.0
    fig.add_trace(go.Scatter(x=t2, y=sub2['x'], name='X', line=dict(color='blue'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t2, y=sub2['y'], name='Y', line=dict(color='orange'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t2, y=sub2['z'], name='Z', line=dict(color='green'), showlegend=False), row=2, col=1)
else:
    fig.add_annotation(text="No data for this user/activity", row=2, col=1)

# Layout settings (Force same Y-axis scale for scientific accuracy)
fig.update_layout(
    height=700,
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=40, b=20)
)
fig.update_yaxes(title_text="Acceleration (g)", range=[-22, 22], row=1, col=1)
fig.update_yaxes(title_text="Acceleration (g)", range=[-22, 22], row=2, col=1)
fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# 5. Summary Statistics Section
st.markdown("### 📊 Summary Statistics (Current Selection)")
col1, col2 = st.columns(2)

with col1:
    st.write(f"**{act1} - User {user1}** (First {num_samples} samples)")
    if not sub1.empty:
        st.dataframe(sub1[['x', 'y', 'z']].describe().loc[['mean', 'std', 'min', 'max']].round(3))

with col2:
    st.write(f"**{act2} - User {user2}** (First {num_samples} samples)")
    if not sub2.empty:
        st.dataframe(sub2[['x', 'y', 'z']].describe().loc[['mean', 'std', 'min', 'max']].round(3))