"""
Interactive Traffic Simulation Dashboard
Run with: streamlit run dashboard.py
"""

import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import matplotlib.patches as mpatches
from src.simulation.simulator import Simulator, SimulationConfig, ControllerType
from src.simulation.traffic_generator import TrafficScenario
from src.controllers.dl_controller import DeepLearningController
import os
import threading
from queue import Queue
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="AI Traffic Control Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'current_controller' not in st.session_state:
    st.session_state.current_controller = "Fixed"
if 'current_scenario' not in st.session_state:
    st.session_state.current_scenario = "Normal"
if 'simulator' not in st.session_state:
    st.session_state.simulator = None
if 'queue_history' not in st.session_state:
    st.session_state.queue_history = []
if 'vehicle_history' not in st.session_state:
    st.session_state.vehicle_history = []

# Controller options
CONTROLLERS = {
    "Fixed": ControllerType.FIXED,
    "Rule-Based": ControllerType.RULE_BASED,
    "Deep Learning": ControllerType.DEEP_LEARNING
}

SCENARIOS = {
    "Normal": TrafficScenario.NORMAL,
    "Morning Rush": TrafficScenario.MORNING_RUSH,
    "Evening Rush": TrafficScenario.EVENING_RUSH,
    "Balanced": TrafficScenario.BALANCED,
    "Heavy NS": TrafficScenario.HEAVY_NS,
    "Heavy EW": TrafficScenario.HEAVY_EW,
    "Sudden Spike": TrafficScenario.SUDDEN_SPIKE,
    "Low Traffic": TrafficScenario.LOW_TRAFFIC
}


def create_intersection_plot(queues, phase, elapsed):
    """Create a matplotlib visualization of the intersection"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Draw intersection background
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw roads
    ax.add_patch(Rectangle((-2, -5), 4, 10, facecolor='#2c3e50', alpha=0.3))
    ax.add_patch(Rectangle((-5, -2), 10, 4, facecolor='#2c3e50', alpha=0.3))
    
    # Intersection center
    center = Circle((0, 0), 1.5, facecolor='white', edgecolor='#2c3e50', linewidth=2)
    ax.add_patch(center)
    
    # Draw traffic lights
    green_roads = ['N', 'S'] if phase == 'NS_GO' else ['E', 'W']
    
    positions = {
        'N': (0, 3.5),
        'S': (0, -3.5),
        'E': (3.5, 0),
        'W': (-3.5, 0)
    }
    
    for road, pos in positions.items():
        color = '#2ecc71' if road in green_roads else '#e74c3c'
        circle = Circle(pos, 0.4, facecolor=color, edgecolor='#2c3e50', linewidth=2)
        ax.add_patch(circle)
        
        # Add queue count
        count = queues.get(road, 0)
        ax.text(pos[0], pos[1] - 1.2, f'🚗 {count}', fontsize=10, ha='center', va='center')
    
    # Add labels
    ax.text(0, 4.8, 'NORTH', fontsize=12, fontweight='bold', ha='center')
    ax.text(0, -4.8, 'SOUTH', fontsize=12, fontweight='bold', ha='center')
    ax.text(4.8, 0, 'EAST', fontsize=12, fontweight='bold', ha='center')
    ax.text(-4.8, 0, 'WEST', fontsize=12, fontweight='bold', ha='center')
    
    # Phase info
    ax.text(0, 4.2, f'Phase: {phase}', fontsize=10, ha='center', style='italic')
    ax.text(0, -4.2, f'Elapsed: {elapsed:.1f}s', fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    return fig


def run_simulation_async(controller_type, scenario, duration=30):
    """Run simulation in background thread"""
    try:
        config = SimulationConfig(
            duration=duration,
            scenario=SCENARIOS[scenario],
            controller=CONTROLLERS[controller_type],
            seed=42,
            fixed_green_time=20.0,
            rule_min_green=8.0,
            rule_max_green=30.0,
            rule_hysteresis=0.2,
            verbose=False
        )
        
        simulator = Simulator(config)
        
        # Store for live updates
        st.session_state.simulator = simulator
        
        # Run simulation
        result = simulator.run()
        
        # Store result
        st.session_state.results = result
        st.session_state.simulation_running = False
        
        # Collect history for visualization
        if hasattr(simulator, 'queue_history'):
            st.session_state.queue_history = simulator.queue_history
        
        return result
        
    except Exception as e:
        st.error(f"Simulation error: {e}")
        st.session_state.simulation_running = False
        return None


# ============= MAIN DASHBOARD =============

# Header
st.markdown('<div class="main-header">🚦 AI-Based Adaptive Traffic Signal Optimization</div>', 
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("🎛️ Controls")
    
    # Controller selection
    st.subheader("Controller")
    controller = st.radio(
        "Select Controller",
        ["Fixed", "Rule-Based", "Deep Learning"],
        index=0,
        help="Choose which traffic controller to simulate"
    )
    st.session_state.current_controller = controller
    
    # Scenario selection
    st.subheader("Traffic Scenario")
    scenario = st.selectbox(
        "Select Scenario",
        list(SCENARIOS.keys()),
        index=0
    )
    st.session_state.current_scenario = scenario
    
    # Duration
    st.subheader("Simulation Settings")
    duration = st.slider(
        "Simulation Duration (seconds)",
        min_value=10,
        max_value=120,
        value=30,
        step=5
    )
    
    # Run button
    st.markdown("---")
    run_button = st.button(
        "▶️ Run Simulation",
        type="primary",
        use_container_width=True
    )
    
    # Status
    st.markdown("---")
    st.subheader("Status")
    status_placeholder = st.empty()
    
    if st.session_state.simulation_running:
        status_placeholder.info("🔄 Simulation running...")
    else:
        status_placeholder.success("✅ Ready")
    
    # Controller info
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.caption("""
    **Fixed**: Fixed green times (20s each)
    
    **Rule-Based**: Adaptive switching based on queue lengths
    
    **Deep Learning**: AI-powered decisions using trained neural network
    """)

# Main content area
col1, col2, col3 = st.columns([2, 1, 1])

# Run simulation when button clicked
if run_button:
    st.session_state.simulation_running = True
    st.session_state.results = None
    st.session_state.queue_history = []
    st.session_state.vehicle_history = []
    
    with st.spinner(f"Running {controller} controller with {scenario} scenario..."):
        # Run simulation (synchronous for simplicity)
        try:
            config = SimulationConfig(
                duration=duration,
                scenario=SCENARIOS[scenario],
                controller=CONTROLLERS[controller],
                seed=42,
                fixed_green_time=20.0,
                rule_min_green=8.0,
                rule_max_green=30.0,
                rule_hysteresis=0.2,
                verbose=False
            )
            
            simulator = Simulator(config)
            
            # Run step by step for live updates
            # Simplified: run full simulation
            result = simulator.run()
            st.session_state.results = result
            
            # Store queue history from simulator
            if hasattr(simulator, 'queue_history'):
                st.session_state.queue_history = simulator.queue_history
            
            st.session_state.simulation_running = False
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.simulation_running = False

# Display live simulation if running or results available
if st.session_state.results is not None:
    result = st.session_state.results
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🚗 Vehicles Served", f"{result.vehicles_served}")
    with col2:
        st.metric("⏱️ Avg Wait", f"{result.average_waiting_time:.2f}s")
    with col3:
        st.metric("📏 Avg Queue", f"{result.average_queue_length:.2f}")
    with col4:
        st.metric("📈 Throughput", f"{result.throughput:.0f} veh/h")
    with col5:
        st.metric("🔄 Phase Changes", f"{result.phase_changes}")
    
    # Two columns for visualization
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.subheader("🚦 Intersection Visualization")
        
        # Get current state for visualization
        queues = result.config.scenario.value  # placeholder
        # Create intersection plot with actual data
        fig = create_intersection_plot(
            result.queues if hasattr(result, 'queues') else {'N': 5, 'S': 3, 'E': 7, 'W': 2},
            result.config.controller.value if hasattr(result.config, 'controller') else 'NS_GO',
            15.0
        )
        st.pyplot(fig)
    
    with viz_col2:
        st.subheader("📊 Queue Length History")
        
        # Plot queue history if available
        if st.session_state.queue_history:
            df = pd.DataFrame(st.session_state.queue_history)
            if 'total' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(df['time'], df['total'], linewidth=2)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Queue Length')
                ax.set_title('Total Queue Length Over Time')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        else:
            st.info("Queue history data not available")

else:
    # Show placeholder when no results
    st.info("👈 Select a controller and scenario, then click 'Run Simulation'")
    
    # Display a sample visualization
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚦 Intersection Preview")
        fig = create_intersection_plot(
            {'N': 0, 'S': 0, 'E': 0, 'W': 0},
            'NS_GO',
            0.0
        )
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Expected Results")
        st.markdown("""
        After running the simulation, you'll see:
        
        - **Vehicles Served**: Total vehicles passing through
        - **Average Wait Time**: Time vehicles spend waiting
        - **Average Queue Length**: Vehicles waiting at signals
        - **Throughput**: Vehicles per hour
        - **Phase Changes**: Number of signal switches
        """)

# Comparison section (bottom)
st.markdown("---")
st.subheader("📊 Controller Comparison")

if st.session_state.results is not None:
    # Run all three controllers for comparison
    with st.expander("🔄 Run All Controllers for Comparison"):
        if st.button("⚡ Compare All Controllers", use_container_width=True):
            with st.spinner("Running all controllers..."):
                comparison_results = {}
                progress_bar = st.progress(0)
                
                for i, controller_name in enumerate(["Fixed", "Rule-Based", "Deep Learning"]):
                    config = SimulationConfig(
                        duration=30,
                        scenario=SCENARIOS[scenario],
                        controller=CONTROLLERS[controller_name],
                        seed=42,
                        fixed_green_time=20.0,
                        rule_min_green=8.0,
                        rule_max_green=30.0,
                        verbose=False
                    )
                    
                    simulator = Simulator(config)
                    result = simulator.run()
                    comparison_results[controller_name] = result
                    progress_bar.progress((i + 1) / 3)
                
                # Display comparison
                comparison_df = pd.DataFrame({
                    'Controller': list(comparison_results.keys()),
                    'Vehicles Served': [r.vehicles_served for r in comparison_results.values()],
                    'Avg Wait (s)': [r.average_waiting_time for r in comparison_results.values()],
                    'Avg Queue': [r.average_queue_length for r in comparison_results.values()],
                    'Throughput': [r.throughput for r in comparison_results.values()],
                    'Phase Changes': [r.phase_changes for r in comparison_results.values()]
                })
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Create bar chart comparison
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Vehicles served
                axes[0].bar(comparison_df['Controller'], comparison_df['Vehicles Served'], 
                            color=['#e74c3c', '#3498db', '#2ecc71'])
                axes[0].set_title('Vehicles Served')
                axes[0].grid(True, alpha=0.3)
                
                # Avg wait time
                axes[1].bar(comparison_df['Controller'], comparison_df['Avg Wait (s)'],
                            color=['#e74c3c', '#3498db', '#2ecc71'])
                axes[1].set_title('Avg Wait Time (s)')
                axes[1].grid(True, alpha=0.3)
                
                # Throughput
                axes[2].bar(comparison_df['Controller'], comparison_df['Throughput'],
                            color=['#e74c3c', '#3498db', '#2ecc71'])
                axes[2].set_title('Throughput (veh/h)')
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("🚦 AI-Based Adaptive Traffic Signal Optimization | Deep Learning Course Project")

# Debug info (optional)
if st.checkbox("Show Debug Info"):
    st.json({
        "controller": st.session_state.get('current_controller'),
        "scenario": st.session_state.get('current_scenario'),
        "running": st.session_state.get('simulation_running'),
        "has_results": st.session_state.get('results') is not None
    })