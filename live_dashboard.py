"""
LIVE Interactive Traffic Simulation Dashboard
Shows real-time simulation with animated vehicles
Run with: streamlit run live_dashboard.py
"""

import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
from src.simulation.intersection import Intersection, SignalPhase
from src.simulation.traffic_generator import TrafficGenerator, TrafficScenario
from src.controllers.fixed_controller import FixedController
from src.controllers.rule_controller import RuleBasedController
from src.controllers.dl_controller import DeepLearningController

# Page config
st.set_page_config(
    page_title="🚦 Live Traffic AI Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better look
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: white;
        text-align: center;
        padding: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'simulation_running': False,
        'simulation_complete': False,
        'current_step': 0,
        'max_steps': 120,
        'intersection': None,
        'generator': None,
        'controller': None,
        'controller_type': 'Fixed',
        'scenario': 'Normal',
        'queue_history': [],
        'metrics_history': [],
        'current_metrics': {},
        'vehicles_served_history': [],
        'phase_history': [],
        'waiting_time_history': [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def load_dl_model():
    """Load the trained DL model once"""
    try:
        from tensorflow.keras.models import load_model
        import pickle
        model = load_model('model/best_model.keras')
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        return None, None


def create_controller(intersection, controller_type):
    """Create a controller instance"""
    if controller_type == "Fixed":
        return FixedController(intersection, green_time=20.0)
    elif controller_type == "Rule-Based":
        return RuleBasedController(
            intersection,
            min_green=8.0,
            max_green=30.0,
            hysteresis=0.2
        )
    elif controller_type == "Deep Learning":
        return DeepLearningController(
            intersection,
            model_path='model/best_model.keras',
            scaler_path='model/scaler.pkl',
            min_green=8.0,
            max_green=30.0,
            confidence_threshold=0.4
        )
    return None


def get_scenario_enum(name):
    """Convert scenario name to enum"""
    mapping = {
        "Normal": TrafficScenario.NORMAL,
        "Morning Rush": TrafficScenario.MORNING_RUSH,
        "Evening Rush": TrafficScenario.EVENING_RUSH,
        "Balanced": TrafficScenario.BALANCED,
        "Heavy NS": TrafficScenario.HEAVY_NS,
        "Heavy EW": TrafficScenario.HEAVY_EW,
        "Sudden Spike": TrafficScenario.SUDDEN_SPIKE,
        "Low Traffic": TrafficScenario.LOW_TRAFFIC,
    }
    return mapping.get(name, TrafficScenario.NORMAL)


def initialize_simulation(controller_type, scenario, max_steps):
    """Initialize a new simulation"""
    intersection = Intersection(min_green=8.0, max_green=30.0)
    generator = TrafficGenerator(seed=42)
    generator.set_scenario(get_scenario_enum(scenario))
    generator.apply_scenario_to_intersection(intersection)

    controller = create_controller(intersection, controller_type)

    st.session_state.intersection = intersection
    st.session_state.generator = generator
    st.session_state.controller = controller
    st.session_state.controller_type = controller_type
    st.session_state.scenario = scenario
    st.session_state.current_step = 0
    st.session_state.max_steps = max_steps
    st.session_state.queue_history = []
    st.session_state.metrics_history = []
    st.session_state.vehicles_served_history = []
    st.session_state.phase_history = []
    st.session_state.waiting_time_history = []
    st.session_state.simulation_complete = False
    st.session_state.current_metrics = {}


def run_simulation_step():
    """Run ONE step of the simulation and update all state"""
    if st.session_state.intersection is None:
        return

    intersection = st.session_state.intersection
    generator = st.session_state.generator
    controller = st.session_state.controller
    current_step = st.session_state.current_step

    # Generate vehicles
    new_vehicles = generator.generate_vehicles(float(current_step), 1.0)
    for road, vehicles in new_vehicles.items():
        for vehicle in vehicles:
            intersection.add_vehicle(road, vehicle)

    # Controller decision
    if controller is not None:
        if controller.decide(float(current_step)):
            other_phase = (
                SignalPhase.EW_GO
                if intersection.phase == SignalPhase.NS_GO
                else SignalPhase.NS_GO
            )
            intersection.switch_phase(other_phase, float(current_step))
            if hasattr(controller, 'reset'):
                controller.reset()

    # Step intersection
    intersection.step(float(current_step), 1.0)

    # Collect metrics
    queues = intersection.get_queue_lengths()
    total_queue = sum(queues.values())

    st.session_state.queue_history.append({
        'time': current_step,
        'N': queues['N'],
        'S': queues['S'],
        'E': queues['E'],
        'W': queues['W'],
        'total': total_queue
    })

    st.session_state.vehicles_served_history.append({
        'time': current_step,
        'served': intersection.vehicles_served_total
    })

    st.session_state.waiting_time_history.append({
        'time': current_step,
        'avg_wait': intersection.get_average_waiting_time()
    })

    st.session_state.phase_history.append({
        'time': current_step,
        'phase': intersection.phase.value
    })

    # Throughput must use the CONFIGURED duration, not the current step index.
    # The simulation runs from step 0 to max_steps-1, so current_step is
    # one less than the configured duration at the final step.
    configured_duration = st.session_state.max_steps  # in seconds
    throughput = (intersection.vehicles_served_total / configured_duration) * 3600

    st.session_state.current_metrics = {
        'vehicles_served': intersection.vehicles_served_total,
        'avg_wait': intersection.get_average_waiting_time(),
        'total_queue': total_queue,
        'queues': queues,
        'phase': intersection.phase.value,
        'elapsed': current_step - intersection.phase_start_time,
        'phase_changes': intersection.phase_changes,
        'throughput': throughput,
    }

    st.session_state.current_step = current_step + 1

    if st.session_state.current_step >= st.session_state.max_steps:
        st.session_state.simulation_running = False
        st.session_state.simulation_complete = True


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_intersection_figure(queues, phase, elapsed, served, step):
    """Create animated intersection visualization with Plotly"""
    fig = go.Figure()

    # Background
    fig.add_shape(type="rect", x0=-5, y0=-5, x1=5, y1=5,
                  fillcolor="#e8e8e8", line=dict(width=0))

    # Roads
    fig.add_shape(type="rect", x0=-1.2, y0=-5, x1=1.2, y1=5,
                  fillcolor="#555555", line=dict(width=0))
    fig.add_shape(type="rect", x0=-5, y0=-1.2, x1=5, y1=1.2,
                  fillcolor="#555555", line=dict(width=0))

    # Intersection center
    fig.add_shape(type="rect", x0=-1.2, y0=-1.2, x1=1.2, y1=1.2,
                  fillcolor="#666666", line=dict(color="white", width=2))

    # Road center lines (dashed)
    fig.add_shape(type="line", x0=0, y0=-5, x1=0, y1=-1.2,
                  line=dict(color="yellow", width=2, dash="dash"))
    fig.add_shape(type="line", x0=0, y0=1.2, x1=0, y1=5,
                  line=dict(color="yellow", width=2, dash="dash"))
    fig.add_shape(type="line", x0=-5, y0=0, x1=-1.2, y1=0,
                  line=dict(color="yellow", width=2, dash="dash"))
    fig.add_shape(type="line", x0=1.2, y0=0, x1=5, y1=0,
                  line=dict(color="yellow", width=2, dash="dash"))

    # Traffic lights
    green_roads = ['N', 'S'] if phase == 'NS_GO' else ['E', 'W']

    light_positions = {
        'N': (1.6, 1.6),
        'S': (-1.6, -1.6),
        'E': (1.6, -1.6),
        'W': (-1.6, 1.6),
    }

    for road, (x, y) in light_positions.items():
        color = "#00ff00" if road in green_roads else "#ff0000"
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(size=30, color=color,
                        line=dict(color='white', width=3)),
            name=f'{road} Light',
            showlegend=False,
            hoverinfo='skip'
        ))

    # Vehicle queues (as colored dots)
    road_offsets = {
        'N': (0, 1),
        'S': (0, -1),
        'E': (1, 0),
        'W': (-1, 0),
    }

    for road, count in queues.items():
        dx, dy = road_offsets[road]
        display_count = min(count, 15)  # Show max 15 vehicles per road

        for i in range(display_count):
            # Position vehicles along the road
            distance = 1.5 + i * 0.25

            if road == 'N':
                x, y = dx, dy * distance * 2
            elif road == 'S':
                x, y = dx, -dy * distance * 2 if dy < 0 else dy * distance * 2
                y = -1.5 - i * 0.25
                x = 0.5
            elif road == 'E':
                x = 1.5 + i * 0.25
                y = -0.5
            elif road == 'W':
                x = -1.5 - i * 0.25
                y = 0.5
            else:
                x, y = 0, 0

            # Color based on whether this road has green
            color = "#2ecc71" if road in green_roads else "#e74c3c"

            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers',
                marker=dict(size=12, color=color,
                            line=dict(color='white', width=1.5)),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Queue count labels
    for road, count in queues.items():
        if road == 'N':
            x, y = -2.5, 3.5
        elif road == 'S':
            x, y = 2.5, -3.5
        elif road == 'E':
            x, y = 3.5, 1.5
        else:
            x, y = -3.5, -1.5

        fig.add_annotation(
            x=x, y=y,
            text=f"<b>{road}</b><br>{count} 🚗",
            showarrow=False,
            font=dict(size=14, color="white"),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="white",
            borderwidth=1,
            borderpad=6
        )

    # Title annotations
    fig.add_annotation(
        x=0, y=4.7,
        text=f"<b>🚦 {phase}</b>",
        showarrow=False,
        font=dict(size=18, color="white"),
        bgcolor="rgba(0,0,0,0.8)",
        bordercolor="white",
        borderwidth=2,
        borderpad=8
    )

    fig.add_annotation(
        x=0, y=-4.7,
        text=f"⏱️ {elapsed:.1f}s | Step {step} | Served: {served}",
        showarrow=False,
        font=dict(size=12, color="white"),
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor="white",
        borderwidth=1,
        borderpad=6
    )

    # Update layout
    fig.update_layout(
        xaxis=dict(range=[-5, 5], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-5, 5], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True,
                   scaleanchor="x", scaleratio=1),
        height=550,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
    )

    return fig


def create_queue_chart(queue_history):
    """Create real-time queue length line chart"""
    if not queue_history:
        return go.Figure()

    df = pd.DataFrame(queue_history)

    fig = go.Figure()

    # Total queue line
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['total'],
        mode='lines',
        name='Total',
        line=dict(color='#e74c3c', width=3),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.2)'
    ))

    # Individual roads
    colors = {'N': '#3498db', 'S': '#2ecc71', 'E': '#f39c12', 'W': '#9b59b6'}
    for road in ['N', 'S', 'E', 'W']:
        if road in df.columns:
            fig.add_trace(go.Scatter(
                x=df['time'], y=df[road],
                mode='lines',
                name=road,
                line=dict(color=colors[road], width=2),
            ))

    fig.update_layout(
        title="📊 Queue Length Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Vehicles in Queue",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1)
    )

    return fig


def create_served_chart(vehicles_served_history):
    """Create vehicles served chart"""
    if not vehicles_served_history:
        return go.Figure()

    df = pd.DataFrame(vehicles_served_history)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['served'],
        mode='lines+markers',
        name='Vehicles Served',
        line=dict(color='#2ecc71', width=3),
        marker=dict(size=4)
    ))

    fig.update_layout(
        title="✅ Vehicles Served Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Vehicles",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def create_waiting_time_chart(waiting_time_history):
    """Create waiting time chart"""
    if not waiting_time_history:
        return go.Figure()

    df = pd.DataFrame(waiting_time_history)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['avg_wait'],
        mode='lines',
        name='Avg Wait',
        line=dict(color='#f39c12', width=3),
        fill='tozeroy',
        fillcolor='rgba(243, 156, 18, 0.2)'
    ))

    fig.update_layout(
        title="⏱️ Average Waiting Time",
        xaxis_title="Time (s)",
        yaxis_title="Seconds",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


# ============================================================
# MAIN DASHBOARD LAYOUT
# ============================================================

st.markdown('<div class="main-header">🚦 Live AI Traffic Control Dashboard</div>',
            unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")

    controller_type = st.selectbox(
        "🎮 Select Controller",
        ["Fixed", "Rule-Based", "Deep Learning"],
        index=0,
        help="Choose which traffic controller to run"
    )

    scenario = st.selectbox(
        "🎭 Traffic Scenario",
        ["Normal", "Morning Rush", "Evening Rush", "Balanced",
         "Heavy NS", "Heavy EW", "Sudden Spike", "Low Traffic"],
        index=0
    )

    max_steps = st.slider(
        "⏱️ Duration (seconds)",
        min_value=30, max_value=180, value=60, step=10
    )

    update_speed = st.slider(
        "⚡ Animation Speed (seconds per step)",
        min_value=0.05, max_value=0.5, value=0.15, step=0.05
    )

    st.markdown("---")

    # Start button
    if st.button("▶️ Start Simulation", use_container_width=True):
        initialize_simulation(controller_type, scenario, max_steps)
        st.session_state.simulation_running = True
        st.rerun()

    # Reset button
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.simulation_running = False
        st.session_state.simulation_complete = False
        st.session_state.intersection = None
        st.session_state.current_step = 0
        st.rerun()

    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("""
    **Fixed**: Fixed 20s green time

    **Rule-Based**: Adaptive based on queues

    **Deep Learning**: AI-based decisions
    """)

# Main content
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### 🚦 Live Intersection View")
    intersection_viz = st.empty()

with col_right:
    st.markdown("### 📊 Live Metrics")

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        served_metric = st.empty()
        queue_metric = st.empty()
    with metric_col2:
        wait_metric = st.empty()
        phase_metric = st.empty()

    st.markdown("---")
    progress_bar = st.empty()

# Bottom charts
st.markdown("---")
st.markdown("### 📈 Live Charts")

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    queue_chart_placeholder = st.empty()

with chart_col2:
    served_chart_placeholder = st.empty()

chart_col3, chart_col4 = st.columns(2)
with chart_col3:
    waiting_chart_placeholder = st.empty()

with chart_col4:
    st.markdown("### 🎯 Results Summary")
    results_placeholder = st.empty()


# ============================================================
# SIMULATION LOOP
# ============================================================

if st.session_state.simulation_running and st.session_state.intersection is not None:
    # Run simulation with live updates
    while st.session_state.current_step < st.session_state.max_steps:

        # Run ONE step
        run_simulation_step()

        # Get current state
        intersection = st.session_state.intersection
        queues = intersection.get_queue_lengths()
        metrics = st.session_state.current_metrics

        # Update intersection visualization
        fig_intersection = create_intersection_figure(
            queues=queues,
            phase=metrics.get('phase', 'NS_GO'),
            elapsed=metrics.get('elapsed', 0),
            served=metrics.get('vehicles_served', 0),
            step=st.session_state.current_step
        )
        intersection_viz.plotly_chart(
            fig_intersection,
            use_container_width=True,
            key=f"intersection_{st.session_state.current_step}"
        )

        # Update metrics cards
        served_metric.metric(
            "🚗 Vehicles Served",
            metrics.get('vehicles_served', 0)
        )
        queue_metric.metric(
            "📏 Total Queue",
            metrics.get('total_queue', 0)
        )
        wait_metric.metric(
            "⏱️ Avg Wait",
            f"{metrics.get('avg_wait', 0):.2f}s"
        )
        phase_metric.metric(
            "🚦 Phase",
            metrics.get('phase', 'NS_GO')
        )

        # Update progress bar
        progress = st.session_state.current_step / st.session_state.max_steps
        progress_bar.progress(
            progress,
            text=f"Step {st.session_state.current_step} / {st.session_state.max_steps}"
        )

        # Update charts (every 5 steps to reduce flicker)
        if st.session_state.current_step % 5 == 0 or st.session_state.current_step >= st.session_state.max_steps:
            fig_queue = create_queue_chart(st.session_state.queue_history)
            queue_chart_placeholder.plotly_chart(
                fig_queue,
                use_container_width=True,
                key=f"queue_{st.session_state.current_step}"
            )

            fig_served = create_served_chart(st.session_state.vehicles_served_history)
            served_chart_placeholder.plotly_chart(
                fig_served,
                use_container_width=True,
                key=f"served_{st.session_state.current_step}"
            )

            fig_wait = create_waiting_time_chart(st.session_state.waiting_time_history)
            waiting_chart_placeholder.plotly_chart(
                fig_wait,
                use_container_width=True,
                key=f"wait_{st.session_state.current_step}"
            )

        # Sleep for animation speed
        time.sleep(update_speed)

    # Simulation complete
    st.session_state.simulation_running = False
    st.session_state.simulation_complete = True

    # Show final results
    intersection = st.session_state.intersection
    final_metrics = st.session_state.current_metrics

    with results_placeholder.container():
        st.success("✅ Simulation Complete!")
        st.metric("Vehicles Served", final_metrics.get('vehicles_served', 0))
        st.metric("Avg Wait Time", f"{final_metrics.get('avg_wait', 0):.2f}s")
        st.metric("Phase Changes", final_metrics.get('phase_changes', 0))
        st.metric("Throughput", f"{final_metrics.get('throughput', 0):.0f} veh/h")

    # Celebration
    st.balloons()

elif st.session_state.simulation_complete and st.session_state.intersection is not None:
    # Show completed state
    intersection = st.session_state.intersection
    queues = intersection.get_queue_lengths()
    metrics = st.session_state.current_metrics

    fig_intersection = create_intersection_figure(
        queues=queues,
        phase=metrics.get('phase', 'NS_GO'),
        elapsed=metrics.get('elapsed', 0),
        served=metrics.get('vehicles_served', 0),
        step=st.session_state.current_step
    )
    intersection_viz.plotly_chart(fig_intersection, use_container_width=True)

    served_metric.metric("🚗 Vehicles Served", metrics.get('vehicles_served', 0))
    queue_metric.metric("📏 Total Queue", metrics.get('total_queue', 0))
    wait_metric.metric("⏱️ Avg Wait", f"{metrics.get('avg_wait', 0):.2f}s")
    phase_metric.metric("🚦 Phase", metrics.get('phase', 'NS_GO'))

    progress_bar.progress(1.0, text="✅ Complete")

    if st.session_state.queue_history:
        queue_chart_placeholder.plotly_chart(
            create_queue_chart(st.session_state.queue_history),
            use_container_width=True
        )
        served_chart_placeholder.plotly_chart(
            create_served_chart(st.session_state.vehicles_served_history),
            use_container_width=True
        )
        waiting_chart_placeholder.plotly_chart(
            create_waiting_time_chart(st.session_state.waiting_time_history),
            use_container_width=True
        )

    with results_placeholder.container():
        st.success("✅ Simulation Complete!")
        st.metric("Vehicles Served", metrics.get('vehicles_served', 0))
        st.metric("Avg Wait Time", f"{metrics.get('avg_wait', 0):.2f}s")
        st.metric("Phase Changes", metrics.get('phase_changes', 0))
        st.metric("Throughput", f"{metrics.get('throughput', 0):.0f} veh/h")

else:
    # Initial state - show empty view
    fig_empty = create_intersection_figure(
        queues={'N': 0, 'S': 0, 'E': 0, 'W': 0},
        phase='NS_GO',
        elapsed=0,
        served=0,
        step=0
    )
    intersection_viz.plotly_chart(fig_empty, use_container_width=True)

    served_metric.metric("🚗 Vehicles Served", 0)
    queue_metric.metric("📏 Total Queue", 0)
    wait_metric.metric("⏱️ Avg Wait", "0.00s")
    phase_metric.metric("🚦 Phase", "NS_GO")

    progress_bar.progress(0, text="Ready to start")

    st.info("👈 Click **Start Simulation** in the sidebar to begin!")