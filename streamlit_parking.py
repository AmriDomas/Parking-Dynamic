# app.py (fixed)
import streamlit as st

# ===== MUST be the first Streamlit command =====
st.set_page_config(page_title="Parking AI: Forecast, Queue & Pricing", layout="wide")
st.title("🚗 Parking AI – Occupancy • Queue • Dynamic Pricing")
# =================================================

import os
import io
import json
import tempfile
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import requests
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pytz
import holidays


# ====== Load model occup ======
@st.cache_resource
def load_model_occup():
    return joblib.load("best_model_occup.pkl")

# ====== Load model queue ======
@st.cache_resource
def load_model_queue():
    return joblib.load("model_queue.pkl")

# ====== Load dataset ======
@st.cache_data
def load_data():
    return pd.read_csv("parkingStream.csv")  # ganti kalau nama file beda

# ====== Prepare data & encoders ======
df_raw = load_data().copy()
df = load_data().copy()
# Function to load pickled models or encoders from local file or URL
def preprocess_parking_data(df: pd.DataFrame, random_seed: int = 101) -> pd.DataFrame:
    # Convert timestamp ke WIB
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    wib = pytz.timezone("Asia/Jakarta")
    df['Timestamp_WIB'] = df['Timestamp'].dt.tz_localize('UTC').dt.tz_convert(wib)

    # Extract waktu
    df['Hour'] = df['Timestamp_WIB'].dt.hour
    df['DayOfWeek'] = df['Timestamp_WIB'].dt.dayofweek
    df['DayName_text'] = df['Timestamp_WIB'].dt.day_name()
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    # Holiday flag
    id_holidays = holidays.country_holidays("ID")
    df['IsHoliday'] = df['Timestamp_WIB'].dt.date.astype(str).isin(
        [str(d) for d in id_holidays]
    ).astype(int)

    # Time category
    def map_time_category(hour):
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"
    df['TimeCategory'] = df['Hour'].apply(map_time_category)

    # Exit timestamp tetap
    np.random.seed(random_seed)
    df['Exit_Timestamp_WIB'] = df['Timestamp_WIB'] + pd.to_timedelta(
        np.random.randint(30, 300, size=len(df)), unit='m'
    )

    # Duration
    df['Duration_Minutes'] = (
        df['Exit_Timestamp_WIB'] - df['Timestamp_WIB']
    ).dt.total_seconds() / 60

    # Rata-rata durasi
    avg_duration = df.groupby(
        ['TimeCategory', 'IsWeekend', 'IsHoliday']
    )['Duration_Minutes'].mean().reset_index()
    avg_duration.rename(columns={'Duration_Minutes': 'AvgDuration_Minutes'}, inplace=True)
    df = df.merge(avg_duration, on=['TimeCategory', 'IsWeekend', 'IsHoliday'], how='left')
    df['EstimatedDuration_Minutes'] = df['AvgDuration_Minutes'].round()

    # Special day flag
    df['IsSpecialDay_Flag'] = df['IsSpecialDay'].astype(int)

    # Mappings
    vehicle_map = {'car': 0, 'bike': 1, 'cycle': 2, 'truck': 3}
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
               'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    time_map = {'Evening': 0, 'Night': 1, 'Afternoon': 2, 'Morning': 3}
    system_code_map = {code: idx for idx, code in enumerate(df['SystemCodeNumber'].unique())}
    traffic_map = {'low': 0, 'medium': 1, 'high': 2}

    df['SystemCodeNumber'] = df['SystemCodeNumber'].map(system_code_map).astype(int)
    df['VehicleType'] = df['VehicleType'].map(vehicle_map).astype(int)
    df['DayName'] = df['DayName_text'].map(day_map).astype(int)
    df['TimeCategory'] = df['TimeCategory'].map(time_map).astype(int)
    df['TrafficConditionNearby'] = df['TrafficConditionNearby'].map(traffic_map)

    # Encode cyclical features
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek']/7)

    # Queue bin
    def queue_bin(q):
        if q <= 2:
            return 'low'
        elif q <= 6:
            return 'medium'
        else:
            return 'high'
    queue_bin_map = {'low': 0, 'medium': 1, 'high': 2}

    if 'QueueLength' in df.columns:
        df['QueueBin'] = df['QueueLength'].apply(queue_bin).map(queue_bin_map)
    else:
        # kalau tidak ada QueueLength sama sekali, default semua low
        df['QueueBin'] = 0

    return df

df_train = preprocess_parking_data(df)

# Choose feature set that does NOT include Occupancy (we predict queue or occupancy from scratch)
feature_cols = ['SystemCodeNumber', 'Capacity', 'DayName', 
             'VehicleType', 'TrafficConditionNearby', 'IsSpecialDay',
             'Hour', 'DayOfWeek', 'IsWeekend', 'IsHoliday', 'TimeCategory',
             'EstimatedDuration_Minutes', 'IsSpecialDay_Flag',
             'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos']

X = df_train[feature_cols].copy()

# ====== Sidebar & Navigation ======
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Select Menu", ["Analysis", "Prediction"])

# ====== ANALYSIS ======
if menu == "Analysis":
    st.title("📊 Data Analysis - Parking Dynamic")
    df = df.copy()
    
    # CSS untuk tab biar rata
    st.markdown("""
    <style>
    /* Tab container rata */
    div[data-baseweb="tab-list"] {
        justify-content: space-between !important;
    }

    /* Ukuran font tab 10px */
    div[data-baseweb="tab"] > button {
        font-size: 20px !important;
        text-align: center !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 4px;
        padding: 6px 10px !important;
        line-height: 1.2 !important;
    }

    /* Tab aktif */
    div[data-baseweb="tab"][aria-selected="true"] > button {
        background-color: #1976d2 !important;
        color: white !important;
        border-radius: 6px 6px 0 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Tab setup
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Preview & Stats", "📈 Visualization", "🔥 Correlation", "🎨 Custom Plot", "📈 Occupancy & Turnover Analysis"
    ])

    def get_dynamic_palette(n):
            """Generate palette: if n>3 use darkening gradient, else fixed Set2."""
            if n <= 3:
                return sns.color_palette("Set2", n)
            else:
                # gradasi dari terang ke gelap (Blues)
                return sns.color_palette("Blues", n)

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df_raw.head(10), use_container_width=True)

        st.subheader("Basic Statistics")
        st.write(df_raw.describe(include='all'), use_container_width=True)


    def plot_categorical(df_raw, col):
        num_cat = df_raw[col].nunique(dropna=False)
        palette = get_dynamic_palette(num_cat)

        fig, ax = plt.subplots(figsize=(6, 3))
        order = df_raw[col].value_counts().index
        sns.countplot(x=col, data=df_raw, order=order, palette=palette, ax=ax)

        # Title & ticks lebih proporsional
        ax.set_title(f"Distribusi {col}", fontsize=8, weight='bold')
        ax.set_xlabel(col, fontsize=6)  # benerin param
        ax.set_ylabel("Count", fontsize=6)
        ax.tick_params(axis='x', rotation=45, labelsize=5)
        ax.tick_params(axis='y', labelsize=5)

        # Label bar lebih kecil, posisinya di edge
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', fontsize=4, label_type='edge', padding=1)

        sns.despine()  # buang border luar
        st.pyplot(fig)


    with tab2:

        st.markdown("""
        <style>
        [data-testid="stDataFrame"] {
            width: 100% !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.subheader("📊 Distribution Category")
        # exclude hanya kolom 'Id' dan 'Timestamp' (persis nama itu)
        exclude_cols = ["Id", "Timestamp"]

        kategori_cols = [c for c in df_raw.select_dtypes(include='object').columns if c not in exclude_cols]

        col_kategori = st.selectbox("Select the category column", kategori_cols)

        col1, col2 = st.columns([1,1])

        with col1:
            st.subheader("📊 Distribusi Kategori (Bar Chart)")
            fig, ax = plt.subplots(figsize=(5, 4))  

            order = df_raw[col_kategori].value_counts().index
            num_cat = len(order)
            palette = get_dynamic_palette(num_cat)

            sns.countplot(
                x=col_kategori, 
                data=df_raw, 
                order=order, 
                palette=palette, 
                width=0.6,  # lebih ramping
                ax=ax
            )

            ax.set_title(f"Distribution {col_kategori}", fontsize=10, weight='bold')
            ax.set_xlabel(col_kategori, fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.tick_params(axis='x', rotation=45, labelsize=7)
            ax.tick_params(axis='y', labelsize=7)

            for container in ax.containers:
                ax.bar_label(container, fmt='%d', fontsize=6, label_type='edge', padding=1)

            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.subheader("📊 Distribusi Kategori (Pie Chart)")
            val_counts = df_raw[col_kategori].value_counts()

            # ukuran fix biar gak berubah-ubah
            fig, ax = plt.subplots(figsize=(5, 4))  

            wedges, texts, autotexts = ax.pie(
                val_counts,
                labels=None,
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette("Blues", len(val_counts))
            )

            # kunci aspect ratio -> pie selalu lingkaran sempurna
            ax.axis('equal')

            percentages = val_counts / val_counts.sum() * 100
            ax.legend(
                wedges,
                [f"{cat} ({p:.1f}%)" for cat, p in zip(val_counts.index, percentages)],
                title=col_kategori,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=6,
                title_fontsize=8
            )
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # --- Occupancy & QueueLength ---
        st.subheader("📊 Mean Occupancy & QueueLength")

        view_type = st.selectbox("View by:", ["Hours", "Days"], index=0, key="peak_view")

        # Group stats dari df_train
        if view_type == "Hours":
            stats = (
                df_train.groupby("Hour")
                .agg({"Occupancy": "mean", "QueueLength": "mean"})
                .reset_index()
            )
            x_col = "Hour"
            order = None  # biarkan urutan 0–23
        else:  # Days
            stats = (
                df_train.groupby("DayName_text")
                .agg({"Occupancy": "mean", "QueueLength": "mean"})
                .reindex(
                    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                )
                .reset_index()
            )
            x_col = "DayName_text"
            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

        # Bikin plot side-by-side
        col3, col4 = st.columns([1,1])

        with col3:
            fig, ax = plt.subplots(figsize=(5,4))
            sns.barplot(x=x_col, y="Occupancy", data=stats, order=order, ax=ax, palette="Blues")
            ax.set_title(f"Mean Occupancy by {view_type}", fontsize=8)
            ax.set_xlabel(view_type, fontsize=6)
            ax.set_ylabel("Mean Occupancy", fontsize=6)
            ax.tick_params(axis="x", rotation=45, labelsize=5)
            ax.tick_params(axis="y", labelsize=5)

            # Label value di atas bar
            for container in ax.containers:
                ax.bar_label(container, fmt="%.1f", fontsize=5, label_type="edge", padding=1)

            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)

        with col4:
            fig, ax = plt.subplots(figsize=(5,4))
            sns.barplot(x=x_col, y="QueueLength", data=stats, order=order, ax=ax, palette="Oranges")
            ax.set_title(f"Mean Queue Length by {view_type}", fontsize=8)
            ax.set_xlabel(view_type, fontsize=6)
            ax.set_ylabel("Mean Queue Length", fontsize=6)
            ax.tick_params(axis="x", rotation=45, labelsize=5)
            ax.tick_params(axis="y", labelsize=7)

            for container in ax.containers:
                ax.bar_label(container, fmt="%.1f", fontsize=5, label_type="edge", padding=1)

            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)



    with tab3:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df_raw.drop(columns=["ID"], errors='ignore').corr(numeric_only=True)
        sns.heatmap(
            corr,
            annot=True,
            cmap="Spectral",
            ax=ax,
            annot_kws={"size": 8}   # angka dalam kotak
        )
        ax.set_title("Correlation Heatmap", fontsize=8)

        # atur ukuran nama kolom & baris
        ax.tick_params(axis='x', labelsize=7, rotation=45)  # label kolom
        ax.tick_params(axis='y', labelsize=7, rotation=0)   # label baris

        # --- Adjust tulisan legend (colorbar) ---
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=8)   # ukuran angka di legend
        cbar.set_label("Correlation Ratio", fontsize=8)  # label legend + fontsize

        st.pyplot(fig)

        # =============================
        # Heatmap Day vs Time Category
        # =============================
        st.subheader("Average Occupancy by Day & Time Category")
        # mapping balik angka → label
        # Mapping balik TimeCategory angka -> teks
        time_map_rev = {0: "Evening", 1: "Night", 2: "Afternoon"}
        df_train['TimeCategory_Label'] = df_train['TimeCategory'].map(time_map_rev)

        # Pivot pakai df_train (DayName_text sudah ada)
        pivot_table = df_train.pivot_table(
            index="DayName_text",
            columns="TimeCategory_Label",
            values="Occupancy",
            aggfunc="mean"
        )

        # Urutan hari & kolom
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        time_order = ["Afternoon", "Evening", "Night"]

        pivot_table = pivot_table.reindex(day_order)
        pivot_table = pivot_table[time_order]

        # Plot heatmap
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            pivot_table,
            annot=True,
            cmap="YlGnBu",
            fmt=".0f",
            ax=ax2,
            annot_kws={"size": 6}  # angka dalam kotak
        )
        ax2.set_title("Average Occupancy by Day & Time Category", fontsize=8)

        # Label sumbu (bisa diubah teks & ukurannya)
        ax2.set_xlabel("Time of Day", fontsize=8, labelpad=8)   # X → TimeCategory
        ax2.set_ylabel("Day of Week", fontsize=8, labelpad=8)   # Y → DayName_text

        ax2.tick_params(axis='x', labelsize=6, rotation=0)
        ax2.tick_params(axis='y', labelsize=6, rotation=0)

        # --- Adjust tulisan legend (colorbar) ---
        cbar = ax2.collections[0].colorbar
        cbar.ax.tick_params(labelsize=7)   # ukuran angka di legend
        cbar.set_label("Occupancy (avg)", fontsize=8)  # label legend + fontsize

        st.pyplot(fig2)

    with tab4:
        st.subheader("📊 Custom Plot (Auto Bar / Scatter / Box)")
        all_cols = [c for c in df_raw.columns if c not in ["ID", "Timestamp"]]
        col_x = st.selectbox("Select Column X", all_cols, index=0)
        col_y = st.selectbox("Select Column Y", all_cols, index=1)

        fig, ax = plt.subplots(figsize=(8, 4))
        x_is_cat = df_raw[col_x].dtype == 'object'
        y_is_cat = df_raw[col_y].dtype == 'object'

        if x_is_cat and y_is_cat:
            num_cat = df_raw[col_y].nunique(dropna=False)
            palette = get_dynamic_palette(num_cat)
            sns.countplot(x=col_x, hue=col_y, data=df_raw, palette=palette, ax=ax)
            ax.set_title(f"Bar Chart: {col_x} vs {col_y}", fontsize=8, weight='bold')
            ax.set_xlabel(col_x, fontsize=6)
            ax.set_ylabel("Count", fontsize=6)
            ax.tick_params(axis='x', rotation=45, labelsize=5)
            ax.tick_params(axis='y', labelsize=5)
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', fontsize=4, label_type='edge', padding=1)

        elif not x_is_cat and not y_is_cat:
            sns.scatterplot(x=col_x, y=col_y, data=df_raw, color="royalblue", ax=ax)
            ax.set_title(f"Scatter Plot: {col_x} vs {col_y}", fontsize=8, weight='bold')
            ax.set_xlabel(col_x, fontsize=6)
            ax.set_ylabel(col_y, fontsize=6)
            ax.tick_params(axis='both', labelsize=5)

        else:
            if x_is_cat:
                num_cat = df_raw[col_x].nunique(dropna=False)
                palette = get_dynamic_palette(num_cat)
                sns.boxplot(x=col_x, y=col_y, data=df_raw, palette=palette, ax=ax)
                ax.set_xlabel(col_x, fontsize=6)
                ax.set_ylabel(col_y, fontsize=6)
            else:
                num_cat = df_raw[col_y].nunique(dropna=False)
                palette = get_dynamic_palette(num_cat)
                sns.boxplot(x=col_y, y=col_x, data=df_raw, palette=palette, ax=ax)
                ax.set_xlabel(col_y, fontsize=6)
                ax.set_ylabel(col_x, fontsize=6)

            ax.set_title(f"Box Plot: {col_x} vs {col_y}", fontsize=8, weight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=5)
            ax.tick_params(axis='y', labelsize=5)

        # Atur legend biar kecil
        leg = ax.get_legend()
        if leg:
            leg.set_title(leg.get_title().get_text(), prop={'size': 5})
            for text in leg.get_texts():
                text.set_fontsize(5)

        sns.despine()
        st.pyplot(fig)
    
    with tab5:
        col1, col2 = st.columns([1,1])

        with col1:
            fig, ax = plt.subplots(figsize=(6,5))
            sns.boxplot(
                data=df_train,
                x='TrafficConditionNearby',
                y='Occupancy',
                hue='IsSpecialDay',
                ax=ax
            )
            ax.set_title("Occupancy vs Traffic vs Special Events", fontsize=8)
            ax.set_xlabel("Traffic Condition", fontsize=6)
            ax.set_ylabel("Occupancy", fontsize=6)
            ax.tick_params(axis="x", labelsize=5)
            ax.tick_params(axis="y", labelsize=5)
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            # Hitung turnover rate

            # Urutan hari manual
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

            # Tambahin kolom DayName_text dari angka DayName
            day_map = dict(enumerate(day_order))  # {0:'Monday', 1:'Tuesday', ...}
            df_train["DayName_text"] = df_train["DayName"].map(day_map)

            turnover = (
                df_train.groupby(['DayName_text','VehicleType'])
                .apply(lambda x: len(x) / (x['Duration_Minutes'].mean() / 60), include_groups=False)
                .reset_index(name='Turnover_per_Hour')
            )

            # Reindex biar urut
            turnover["DayName_text"] = pd.Categorical(turnover["DayName_text"], categories=day_order, ordered=True)
            turnover = turnover.sort_values("DayName_text")

            # Mapping VehicleType angka ke label
            vehicle_map = {
                0: "car",
                1: "bike",
                2: "cycle",
                3: "truck"
            }

            # Copy turnover khusus untuk plot
            turnover_plot = turnover.copy()
            turnover_plot["VehicleType"] = turnover_plot["VehicleType"].map(vehicle_map)

            fig, ax = plt.subplots(figsize=(6,5))
            sns.lineplot(
                data=turnover_plot,
                x="DayName_text",
                y="Turnover_per_Hour",
                hue="VehicleType",
                marker="o",
                ax=ax
            )
            ax.set_title("Estimated Turnover per Hour by Day & Vehicle Type", fontsize=8)
            ax.set_xlabel("Day", fontsize=6)
            ax.set_ylabel("Turnover per Hour", fontsize=6)
            ax.tick_params(axis="x", rotation=45, labelsize=5)
            ax.tick_params(axis="y", labelsize=5)
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)

# ====== PREDICTION ======
elif menu == "Prediction":
    st.title("🤖 Parking Prediction")

    # CSS untuk tab biar rata
    st.markdown("""
    <style>
    /* Tab container rata */
    div[data-baseweb="tab-list"] {
        justify-content: space-between !important;
    }

    /* Ukuran font tab 10px */
    div[data-baseweb="tab"] > button {
        font-size: 20px !important;
        text-align: center !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 4px;
        padding: 6px 10px !important;
        line-height: 1.2 !important;
    }

    /* Tab aktif */
    div[data-baseweb="tab"][aria-selected="true"] > button {
        background-color: #1976d2 !important;
        color: white !important;
        border-radius: 6px 6px 0 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Tab setup
    T1, T2, T3 = st.tabs([
        "🔮 Revenue Simulation", "💰 Recall & Business Score", "📈 Metrics & Explain"
    ])

    with T1:
        # --------------------------
        # Revenue Simulation
        # --------------------------
        st.subheader("Revenue Simulation: Flat vs Dynamic Pricing")

        occup_model = load_model_occup()

        # Duration assumption
        use_minutes = st.selectbox("Duration Source", 
                                ["EstimatedDuration_Minutes", "Duration_Minutes", "Assume Constant"], 
                                index=2, key="duration_source_T1")

        if use_minutes == "Assume Constant":
            est_minutes = st.number_input("Assumed Avg Duration (minutes)", 10, 600, 164, key="est_minutes_T1")
            duration_hours = est_minutes / 60.0
            dur_series = pd.Series([duration_hours] * len(df))
        else:
            if use_minutes not in df.columns or not df[use_minutes].notna().any():
                st.error(f"Column {use_minutes} not found or empty – switch to 'Assume Constant'.")
                st.stop()
            dur_series = (df[use_minutes].fillna(df[use_minutes].median()) / 60.0).clip(lower=0.1, upper=24)

        # Base price & demand factor
        base_price = st.number_input("Base Price per Hour (Rp)", min_value=0, value=5000, step=500, key="base_price_T1")
        demand_scale = st.slider("Demand Factor Scale (1 + α·ratio)", 0.0, 2.0, 1.0, 0.05, key="demand_scale_T1")

        # Predict occupancy
        X = X.copy()
        pred_occ = occup_model.predict(X)
        ratio = np.clip(pred_occ / np.clip(df['Capacity'].values, 1, None), 0, 1)

        # Multipliers
        def traffic_mult_val(v):
            if isinstance(v, (int, np.integer)):
                return {2: 1.4, 1: 1.15, 0: 1.0}.get(int(v), 1.0)
            s = str(v).lower()
            if s in ["high", "tinggi"]:
                return 1.4
            if s in ["average", "medium", "sedang"]:
                return 1.15
            return 1.0

        def event_mult_val(flag):
            try:
                return 1.3 if int(flag) == 1 else 1.0
            except Exception:
                return 1.0

        tmults = np.array([traffic_mult_val(v) for v in df.get('TrafficConditionNearby', pd.Series(["low"]).repeat(len(df)))])
        emults = np.array([event_mult_val(v) for v in df.get('IsSpecialDay', pd.Series([0]).repeat(len(df)))])
        dyn_price = base_price * (1.0 + demand_scale * ratio) * tmults * emults
        flat_price = np.full(len(df), base_price)

        # Revenue calc
        occ_actual = df['Occupancy'].values if 'Occupancy' in df.columns else pred_occ
        rev_flat = flat_price * dur_series.values * occ_actual
        rev_dyn = dyn_price * dur_series.values * occ_actual

        flat_total = float(rev_flat.sum())
        dyn_total = float(rev_dyn.sum())
        uplift = (dyn_total - flat_total) / flat_total * 100 if flat_total > 0 else np.nan

        # Display results
        st.success(f"""
        Flat total revenue: Rp {flat_total:,.0f}

        Dynamic total revenue: Rp {dyn_total:,.0f}

        Estimated uplift: {uplift:.2f}%
        """)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(["Flat", "Dynamic"], [flat_total, dyn_total], width=0.3)
        ax.set_ylabel("Revenue (Rp)", fontsize=6)
        ax.set_title(f"Revenue Comparison (Uplift {uplift:.2f}%)", fontsize=8)

        # X ticks ("Flat", "Dynamic") lebih kecil
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

        for bar, v in zip(bars, [flat_total, dyn_total]):
            ax.text(bar.get_x() + bar.get_width()/2, v, f"Rp {v/1e9:.2f}B", 
                    ha='center', va='bottom', fontsize=6)

        st.pyplot(fig)

    with T2:
        st.subheader("Queue Classification – Recall & Business Score")
        queue_model = load_model_queue()

        if queue_model is None:
            st.warning("Queue model not loaded.")
        else:
            try:
                # predict pakai model queue
                y_pred_queue = queue_model.predict(X)

                if 'QueueBin' in df_train.columns:
                    y_true_queue = df_train['QueueBin']

                    # ambil classification report
                    report = classification_report(y_true_queue, y_pred_queue, output_dict=True)
                    df_report = pd.DataFrame(report).transpose()

                    # tampilkan tabel report di streamlit
                    st.dataframe(df_report)

                    # recall per class
                    recall_scores = {int(k): v for k, v in df_report['recall'].items() if str(k).isdigit()}
                    st.write("Recall per class:", recall_scores)

                    # weight bisnis
                    business_weights = {1: 0.4, 2: 0.6}
                    weighted_score = sum(recall_scores.get(k, 0) * business_weights[k] for k in business_weights)
                    st.success(f"Weighted Business Score: {weighted_score*100:.2f}%")

                    # plot recall per class
                    fig, ax = plt.subplots(figsize=(6,4))
                    classes = list(recall_scores.keys())
                    scores = list(recall_scores.values())
                    bars = ax.bar(classes, scores, color=['skyblue','orange','red'], width=0.3)
                    ax.set_ylim(0,1)
                    ax.set_ylabel("Recall", fontsize=6)
                    ax.set_xlabel("Queue Class", fontsize=6)
                    ax.set_title("Recall per Queue Class", fontsize=8)

                    for bar, score in zip(bars, scores):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{score:.2f}",
                                ha='center', va='bottom', fontsize=6)

                    st.pyplot(fig)

                else:
                    st.info("No QueueBin in data; metrics skipped.")

            except Exception as e:
                st.error(f"Error computing queue metrics: {e}")

    with T3:
        st.subheader("Model Metrics & Feature Importance")
        occup_model = load_model_occup()
        queue_model = load_model_queue()

        # =====================
        # Occupancy Regression
        # =====================
        if occup_model is not None and 'Occupancy' in df.columns and df['Occupancy'].notna().any():
            X = X.copy()
            pred_occ = occup_model.predict(X)
            mae = mean_absolute_error(df['Occupancy'], pred_occ)
            mape = (np.abs(df['Occupancy'] - pred_occ) / np.clip(df['Occupancy'], 1e-6, None)).mean() * 100
            rmse = mean_squared_error(df['Occupancy'], pred_occ) ** 0.5
            st.markdown(f"**Occupancy – MAE:** {mae:,.2f} | **MAPE:** {mape:.2f}% | **RMSE:** {rmse:,.2f}")

            # Dua kolom berdampingan
            col1, col2 = st.columns([1,1])

            with col1:
                # Pred vs Actual scatter + line regresi
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.regplot(
                    x=df['Occupancy'].values,
                    y=pred_occ,
                    scatter_kws={'alpha':0.6, 's':20},
                    line_kws={'color':'red'},
                    ax=ax
                )
                max_val = max(df['Occupancy'].max(), pred_occ.max())
                min_val = min(df['Occupancy'].min(), pred_occ.min())
                ax.plot([min_val, max_val], [min_val, max_val], 'g--', label='Perfect Prediction')
                ax.set_xlabel("Actual Occupancy", fontsize=6)
                ax.set_ylabel("Predicted Occupancy", fontsize=6)
                ax.set_title("Actual vs Predicted Occupancy", fontsize=8)
                ax.tick_params(axis='x', labelsize=6)
                ax.tick_params(axis='y', labelsize=6)
                ax.legend(fontsize=6)
                ax.grid(True, linestyle="--", alpha=0.5)
                sns.despine()
                plt.tight_layout()
                st.pyplot(fig)

            with col2:
                if hasattr(occup_model, 'feature_importances_'):
                    fi = pd.Series(occup_model.feature_importances_, index=X.columns).sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.barh(fi.head(15).index[::-1], fi.head(15).values[::-1])
                    ax.set_title("Feature Importance – Occupancy", fontsize=8)
                    ax.tick_params(axis='x', labelsize=6)
                    ax.tick_params(axis='y', labelsize=6)
                    for i, v in enumerate(fi.head(15).values[::-1]):
                        ax.text(v, i, f"{v:.3f}", va='center', ha='left', fontsize=5)
                    sns.despine()
                    plt.tight_layout()
                    st.pyplot(fig)

        else:
            st.info("Provide ground-truth Occupancy to compute regression metrics.")

        # =====================
        # Queue Classification
        # =====================
        if queue_model is not None:
            try:
                if 'QueueBin' not in df_train.columns:
                    st.warning("⚠️ QueueBin tidak ada di dataset, dibuat default (semua=0).")
                    df_train['QueueBin'] = 0

                yq_pred = queue_model.predict(X)

                if df_train['QueueBin'].nunique() > 1:
                    report = classification_report(df_train['QueueBin'], yq_pred, output_dict=True)
                    st.write("**Queue Classification – Accuracy:** {:.2f}%".format(report["accuracy"]*100))
                    rec = {int(k): v["recall"] for k,v in report.items() if str(k).isdigit()}
                    wscore = 0.4*rec.get(1,0) + 0.6*rec.get(2,0)
                    st.write("**Business Impact Score:** {:.2f}%".format(wscore*100))

                    # Dua kolom berdampingan
                    col3, col4 = st.columns([1,1])

                    with col3:
                        cm = confusion_matrix(df_train['QueueBin'], yq_pred, labels=[0,1,2])
                        fig, ax = plt.subplots(figsize=(5, 4))
                        im = ax.imshow(cm, cmap='Blues')
                        ax.set_title('Confusion Matrix – Queue', fontsize=6)
                        ax.set_xlabel('Predicted', fontsize=5)
                        ax.set_ylabel('Actual', fontsize=5)
                        ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
                        ax.set_xticklabels([0,1,2], fontsize=5)
                        ax.set_yticklabels([0,1,2], fontsize=5)
                        for (i,j), v in np.ndenumerate(cm):
                            ax.text(j, i, str(v), ha='center', va='center', fontsize=5)
                        sns.despine()
                        plt.tight_layout()
                        st.pyplot(fig)

                    with col4:
                        if hasattr(queue_model, 'feature_importances_'):
                            fiq = pd.Series(queue_model.feature_importances_, index=X.columns).sort_values(ascending=False)
                            fig, ax = plt.subplots(figsize=(5, 4))
                            ax.barh(fiq.head(15).index[::-1], fiq.head(15).values[::-1])
                            ax.set_title("Feature Importance – Queue", fontsize=8)
                            ax.tick_params(axis='x', labelsize=6)
                            ax.tick_params(axis='y', labelsize=6)
                            for i, v in enumerate(fiq.head(15).values[::-1]):
                                ax.text(v, i, f"{v:.3f}", va='center', ha='left', fontsize=5)
                            sns.despine()
                            plt.tight_layout()
                            st.pyplot(fig)

            except Exception as e:
                st.error(f"Queue Classification Error: {e}")

    
                     


                

