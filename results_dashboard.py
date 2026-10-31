"""
Results Dashboard - Visualize experiment results
Run with: streamlit run results_dashboard.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

st.set_page_config(
    page_title="AI Traffic Control - Results",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Traffic Control Experiment Results")

# Load results
if os.path.exists("results/experiment_results.csv"):
    df = pd.read_csv("results/experiment_results.csv")
    st.success(f"✅ Loaded {len(df)} result rows")
else:
    st.warning("No results found. Run the experiment first!")
    st.stop()

# Sidebar filters
st.sidebar.title("🎯 Filters")
controllers = st.sidebar.multiselect(
    "Select Controllers",
    options=df['controller'].unique(),
    default=df['controller'].unique()
)

scenarios = st.sidebar.multiselect(
    "Select Scenarios",
    options=df['scenario'].unique(),
    default=df['scenario'].unique()
)

# Filter data
filtered_df = df[df['controller'].isin(controllers) & df['scenario'].isin(scenarios)]

# Metrics overview
st.subheader("📈 Performance Metrics Overview")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_served = filtered_df['vehicles_served'].mean()
    st.metric("Avg Vehicles Served", f"{avg_served:.1f}")
with col2:
    avg_wait = filtered_df['avg_wait'].mean()
    st.metric("Avg Waiting Time", f"{avg_wait:.2f}s")
with col3:
    avg_queue = filtered_df['avg_queue'].mean()
    st.metric("Avg Queue Length", f"{avg_queue:.2f}")
with col4:
    avg_throughput = filtered_df['throughput'].mean()
    st.metric("Avg Throughput", f"{avg_throughput:.0f} veh/h")
with col5:
    avg_congestion = filtered_df['congestion'].mean()
    st.metric("Avg Congestion", f"{avg_congestion:.1f}%")

# Comparison charts
st.subheader("📊 Controller Comparison")

# Create comparison charts
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

metrics = [
    ('vehicles_served', 'Vehicles Served', 'Higher is Better', 'blue'),
    ('avg_wait', 'Average Waiting Time (s)', 'Lower is Better', 'red'),
    ('avg_queue', 'Average Queue Length', 'Lower is Better', 'orange'),
    ('throughput', 'Throughput (veh/h)', 'Higher is Better', 'green'),
    ('congestion', 'Congestion (%)', 'Lower is Better', 'purple'),
    ('phase_changes', 'Phase Changes', 'Neutral', 'brown')
]

for idx, (metric, label, note, color) in enumerate(metrics):
    if idx < len(axes):
        ax = axes[idx]
        
        # Group by controller
        controller_means = filtered_df.groupby('controller')[metric].mean()
        
        bars = ax.bar(controller_means.index, controller_means.values, color=['#e74c3c', '#3498db', '#2ecc71'])
        ax.set_title(f'{label}\n({note})')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
st.pyplot(fig)

# Scenario-specific performance
st.subheader("🎭 Performance by Scenario")

# Heatmap of vehicles served
pivot = filtered_df.pivot(index='scenario', columns='controller', values='vehicles_served')

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
ax.set_title('Vehicles Served by Scenario and Controller')
ax.set_xlabel('Controller')
ax.set_ylabel('Scenario')
st.pyplot(fig)

# Detailed table
st.subheader("📋 Detailed Results Table")
st.dataframe(filtered_df, use_container_width=True)

# Download results
st.download_button(
    label="📥 Download Results CSV",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_results.csv",
    mime="text/csv"
)

# Best controller analysis
st.subheader("🏆 Best Controller Analysis")

best_by_scenario = filtered_df.loc[filtered_df.groupby('scenario')['vehicles_served'].idxmax()]
best_by_scenario = best_by_scenario[['scenario', 'controller', 'vehicles_served', 'avg_wait', 'throughput']]
best_by_scenario.columns = ['Scenario', 'Best Controller', 'Vehicles Served', 'Avg Wait (s)', 'Throughput']

st.dataframe(best_by_scenario, use_container_width=True)

# Summary statistics
st.subheader("📊 Summary Statistics")
summary = filtered_df.groupby('controller').agg({
    'vehicles_served': ['mean', 'std'],
    'avg_wait': ['mean', 'std'],
    'throughput': ['mean', 'std'],
    'phase_changes': ['mean', 'std']
}).round(2)

st.dataframe(summary, use_container_width=True)