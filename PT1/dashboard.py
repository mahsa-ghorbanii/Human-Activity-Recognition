
# ============================================
# dashboard.py - Interactive Dashboard for HAR
# Section 2-5: Comparing Three Participants
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


# ============================================
# 1. Page Configuration
# ============================================

st.set_page_config(
    page_title="HAR Dashboard",
    page_icon="🏃",
    layout="wide"
)


# ============================================
# 2. Data Loading
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "WISDM_ar_v1.1_raw.txt")


@st.cache_data
def load_data():

    records = []

    with open(DATA_PATH, "r", encoding="utf-8", errors="ignore") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            for seg in line.split(";"):

                seg = seg.strip().strip(",")

                if not seg:
                    continue

                parts = [
                    p.strip()
                    for p in seg.split(",")
                    if p.strip() != ""
                ]

                if len(parts) == 6:
                    records.append(parts)

    df = pd.DataFrame(
        records,
        columns=[
            "user",
            "activity",
            "timestamp",
            "x",
            "y",
            "z"
        ]
    )

    df["user"] = pd.to_numeric(df["user"], errors="coerce")

    for col in ["timestamp", "x", "y", "z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    df["user"] = df["user"].astype(int)

    df["activity"] = df["activity"].astype(str).str.strip()

    df = df.sort_values(
        ["user", "activity", "timestamp"]
    ).reset_index(drop=True)

    return df


# Load data

with st.spinner("Loading data..."):

    try:
        df = load_data()

    except Exception as e:

        st.error(f"Error loading dataset: {e}")
        st.stop()


if df.empty:

    st.error("Dataset is empty. Please check the data file.")
    st.stop()


# ============================================
# 3. Sidebar - User Interface
# ============================================

st.sidebar.header("⚙️ Settings")

st.sidebar.markdown(
    "Select activities and participants to compare accelerometer signals."
)


activities = sorted(df["activity"].unique())

users = sorted(df["user"].unique())


# --------------------------------------------
# Signal 1
# --------------------------------------------

st.sidebar.subheader("Signal 1")

act1 = st.sidebar.selectbox(
    "Activity 1",
    activities,
    index=activities.index("Walking")
    if "Walking" in activities else 0
)

user1 = st.sidebar.selectbox(
    "User 1",
    users,
    index=users.index(3) if 3 in users else 0
)


# --------------------------------------------
# Signal 2
# --------------------------------------------

st.sidebar.subheader("Signal 2")

act2 = st.sidebar.selectbox(
    "Activity 2",
    activities,
    index=activities.index("Walking")
    if "Walking" in activities else 0
)

user2 = st.sidebar.selectbox(
    "User 2",
    users,
    index=users.index(4)
    if 4 in users
    else min(1, len(users) - 1)
)


# --------------------------------------------
# Signal 3
# --------------------------------------------

st.sidebar.subheader("Signal 3")

act3 = st.sidebar.selectbox(
    "Activity 3",
    activities,
    index=activities.index("Walking")
    if "Walking" in activities else 0
)

user3 = st.sidebar.selectbox(
    "User 3",
    users,
    index=users.index(5)
    if 5 in users
    else min(2, len(users) - 1)
)


# --------------------------------------------
# Number of samples
# --------------------------------------------

num_samples = st.sidebar.slider(
    "Number of samples (20Hz)",
    min_value=51,
    max_value=1000,
    value=200,
    step=50
)


# ============================================
# 4. Main Page
# ============================================

st.title("🏃 Human Activity Recognition Dashboard")

st.markdown(
    "Compare accelerometer signals across three participants "
    "and different physical activities."
)


# ============================================
# 5. Filter Data
# ============================================

def get_data(act, usr):

    subset = df[
        (df["activity"] == act) &
        (df["user"] == usr)
    ]

    return subset.iloc[:num_samples]


sub1 = get_data(act1, user1)

sub2 = get_data(act2, user2)

sub3 = get_data(act3, user3)


# ============================================
# 6. Create Three Subplots
# ============================================

fig = make_subplots(

    rows=3,
    cols=1,

    subplot_titles=(

        f"{act1} - User {user1}",

        f"{act2} - User {user2}",

        f"{act3} - User {user3}"

    ),

    shared_xaxes=True,

    vertical_spacing=0.10
)


# --------------------------------------------
# Plot 1
# --------------------------------------------

if not sub1.empty:

    t1 = np.arange(len(sub1)) / 20.0

    fig.add_trace(

        go.Scatter(
            x=t1,
            y=sub1["x"],
            name="X",
            line=dict(color="blue")
        ),

        row=1,
        col=1
    )

    fig.add_trace(

        go.Scatter(
            x=t1,
            y=sub1["y"],
            name="Y",
            line=dict(color="orange")
        ),

        row=1,
        col=1
    )

    fig.add_trace(

        go.Scatter(
            x=t1,
            y=sub1["z"],
            name="Z",
            line=dict(color="green")
        ),

        row=1,
        col=1
    )

else:

    fig.add_annotation(
        text="No data for this user/activity",
        row=1,
        col=1
    )


# --------------------------------------------
# Plot 2
# --------------------------------------------

if not sub2.empty:

    t2 = np.arange(len(sub2)) / 20.0

    fig.add_trace(

        go.Scatter(
            x=t2,
            y=sub2["x"],
            name="X",
            line=dict(color="blue"),
            showlegend=False
        ),

        row=2,
        col=1
    )

    fig.add_trace(

        go.Scatter(
            x=t2,
            y=sub2["y"],
            name="Y",
            line=dict(color="orange"),
            showlegend=False
        ),

        row=2,
        col=1
    )

    fig.add_trace(

        go.Scatter(
            x=t2,
            y=sub2["z"],
            name="Z",
            line=dict(color="green"),
            showlegend=False
        ),

        row=2,
        col=1
    )

else:

    fig.add_annotation(
        text="No data for this user/activity",
        row=2,
        col=1
    )


# --------------------------------------------
# Plot 3
# --------------------------------------------

if not sub3.empty:

    t3 = np.arange(len(sub3)) / 20.0

    fig.add_trace(

        go.Scatter(
            x=t3,
            y=sub3["x"],
            name="X",
            line=dict(color="blue"),
            showlegend=False
        ),

        row=3,
        col=1
    )

    fig.add_trace(

        go.Scatter(
            x=t3,
            y=sub3["y"],
            name="Y",
            line=dict(color="orange"),
            showlegend=False
        ),

        row=3,
        col=1
    )

    fig.add_trace(

        go.Scatter(
            x=t3,
            y=sub3["z"],
            name="Z",
            line=dict(color="green"),
            showlegend=False
        ),

        row=3,
        col=1
    )

else:

    fig.add_annotation(
        text="No data for this user/activity",
        row=3,
        col=1
    )


# ============================================
# 7. Layout Settings
# ============================================

fig.update_layout(

    height=1000,

    template="plotly_white",

    hovermode="x unified",

    margin=dict(
        l=20,
        r=20,
        t=50,
        b=20
    )

)


# Same Y-axis scale for all three plots

for row in range(1, 4):

    fig.update_yaxes(

        title_text="Acceleration (g)",

        range=[-22, 22],

        row=row,

        col=1

    )


fig.update_xaxes(

    title_text="Time (seconds)",

    row=3,

    col=1

)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================
# 8. Summary Statistics
# ============================================

st.markdown(
    "### 📊 Summary Statistics (Current Selection)"
)


col1, col2, col3 = st.columns(3)


# --------------------------------------------
# Statistics - Signal 1
# --------------------------------------------

with col1:

    st.write(f"**{act1} - User {user1}**")

    if not sub1.empty:

        stats1 = sub1[["x", "y", "z"]].describe()

        st.dataframe(

            stats1.loc[
                ["mean", "std", "min", "max"]
            ].round(3)

        )

    else:

        st.warning("No data available.")


# --------------------------------------------
# Statistics - Signal 2
# --------------------------------------------

with col2:

    st.write(f"**{act2} - User {user2}**")

    if not sub2.empty:

        stats2 = sub2[["x", "y", "z"]].describe()

        st.dataframe(

            stats2.loc[
                ["mean", "std", "min", "max"]
            ].round(3)

        )

    else:

        st.warning("No data available.")


# --------------------------------------------
# Statistics - Signal 3
# --------------------------------------------

with col3:

    st.write(f"**{act3} - User {user3}**")

    if not sub3.empty:

        stats3 = sub3[["x", "y", "z"]].describe()

        st.dataframe(

            stats3.loc[
                ["mean", "std", "min", "max"]
            ].round(3)

        )

    else:

        st.warning("No data available.")


# ============================================
# 9. Footer
# ============================================

st.markdown("---")

st.caption(
    "Human Activity Recognition | WISDM Dataset | "
    "Accelerometer Sampling Rate: 20 Hz"
)
