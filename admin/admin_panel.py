import streamlit as st
import requests
import pandas as pd
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add shared directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
shared_paths = [
    os.path.join(current_dir, '..', 'shared'),
    '/app/shared',
    os.path.join(os.getcwd(), 'shared'),
]

for path in shared_paths:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path):
        sys.path.insert(0, abs_path)
        break

from mibfhe_lib import ZKProofSystem

# Configuration
CLOUD_URL = os.environ.get("CLOUD_URL", "http://localhost:5000")
BENCHMARK_URL = os.environ.get("BENCHMARK_URL", "http://localhost:5001")

st.set_page_config(
    page_title="Admin Panel - VVV Framework",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Admin Panel - Versatile Verification Cloud Framework")
st.markdown("### Backend Monitoring, Logs, and Analytics")

# Sidebar
st.sidebar.header("Configuration")
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard",
    "Backend Logs",
    "Backend Flow",
    "ML Model Processing",
    "Benchmarks"
])

with tab1:
    st.header("System Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Health checks
    try:
        health_resp = requests.get(f"{CLOUD_URL}/health", timeout=5)
        cloud_status = "Online" if health_resp.status_code == 200 else "Offline"
        cloud_healthy = health_resp.status_code == 200
    except:
        cloud_status = "Offline"
        cloud_healthy = False
    
    try:
        bench_resp = requests.get(f"{BENCHMARK_URL}/health", timeout=5)
        bench_status = "Online" if bench_resp.status_code == 200 else "Offline"
    except:
        bench_status = "Offline"
    
    with col1:
        st.metric("Cloud Server", cloud_status)
    
    with col2:
        st.metric("Benchmark Service", bench_status)
    
    # Get statistics
    try:
        stats_resp = requests.get(f"{CLOUD_URL}/logs/stats", timeout=5)
        if stats_resp.status_code == 200:
            stats = stats_resp.json().get('statistics', {})
            total_ops = stats_resp.json().get('total_operations', 0)
            
            with col3:
                st.metric("Total Operations", total_ops)
            
            with col4:
                zk_ops = sum(v for k, v in stats.items() if 'zk' in k.lower())
                st.metric("ZK Verifications", zk_ops)
        else:
            with col3:
                st.metric("Total Operations", "N/A")
            with col4:
                st.metric("ZK Verifications", "N/A")
    except:
        with col3:
            st.metric("Total Operations", "N/A")
        with col4:
            st.metric("ZK Verifications", "N/A")
    
    # Operation statistics chart
    try:
        stats_resp = requests.get(f"{CLOUD_URL}/logs/stats", timeout=5)
        if stats_resp.status_code == 200:
            stats = stats_resp.json().get('statistics', {})
            if stats:
                fig, ax = plt.subplots(figsize=(10, 4))
                operations = list(stats.keys())
                counts = list(stats.values())
                ax.bar(operations, counts, color='steelblue')
                ax.set_xlabel('Operation Type')
                ax.set_ylabel('Count')
                ax.set_title('Operation Statistics')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not load statistics: {e}")

with tab2:
    st.header("Backend Logs")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        log_limit = st.number_input("Log Limit", min_value=10, max_value=1000, value=100, step=10)
        operation_filter = st.selectbox(
            "Filter by Operation",
            ["All", "zk_verification_start", "zk_proof_verification", "zk_commitment_verification",
             "zk_verification_success", "ml_inference_start", "he_operation", "ml_inference_complete"]
        )
    
    try:
        filter_param = None if operation_filter == "All" else operation_filter
        logs_resp = requests.get(
            f"{CLOUD_URL}/logs",
            params={"limit": log_limit, "operation": filter_param},
            timeout=5
        )
        
        if logs_resp.status_code == 200:
            logs_data = logs_resp.json()
            logs = logs_data.get('logs', [])
            
            st.info(f"Showing {len(logs)} of {logs_data.get('total_entries', 0)} total log entries")
            
            # Display logs in expandable sections
            for i, log in enumerate(reversed(logs[-50:])):
                with st.expander(f"{log['operation']} - {log['timestamp']}", expanded=(i < 3)):
                    st.json(log['details'])
        else:
            st.error(f"Failed to fetch logs: {logs_resp.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to cloud server. Is it running?")
    except Exception as e:
        st.error(f"Error: {e}")

with tab3:
    st.header("Backend Processing Flow")
    
    st.markdown("### Privacy Guarantee: No Plaintext Data in Backend")
    st.info("""
    **CRITICAL SECURITY FEATURE:**
    - Backend NEVER receives plaintext names, UIDs, or pincodes
    - Only HASHES (commitments) are stored and verified
    - Backend database contains ONLY SHA-256 hashes
    - Even if backend is compromised, attacker cannot recover original data
    """)
    
    st.markdown("### Complete Request Processing Flow")
    
    # Get recent logs to show flow
    try:
        logs_resp = requests.get(f"{CLOUD_URL}/logs", params={"limit": 100}, timeout=5)
        if logs_resp.status_code == 200:
            logs = logs_resp.json().get('logs', [])
            
            # Group logs by request (using timestamps)
            st.markdown("#### Recent Request Flow")
            
            # Show flow diagram
            st.markdown("""
            ```
            1. Client Request Received
               ├─ Receives: ZK commitment (hash only, NO plaintext)
               ├─ Receives: Encrypted risk factors (encrypted blobs)
               └─ NO plaintext name, UID, or pincode received
               ↓
            2. ZK Proof Verification
               ├─ Extract commitment hash from request
               ├─ Verify proof hash consistency
               └─ Check hash against database (database contains ONLY hashes)
               └─ Result: Verified/Not Verified (without learning identity)
               ↓
            3. Homomorphic Encryption Operations (ML Processing)
               ├─ Receive encrypted inputs: Enc(location), Enc(time), Enc(device), Enc(pincode)
               ├─ Multiply by weights: Enc(x) * weight (homomorphic multiplication)
               ├─ Add weighted terms: Enc(x1) + Enc(x2) (homomorphic addition)
               └─ Add bias: Enc(sum) + bias
               └─ All operations on encrypted data - NO decryption
               ↓
            4. ML Inference Complete
               └─ Return encrypted result (client decrypts)
               └─ Backend NEVER sees plaintext risk scores
            ```
            """)
            
            st.markdown("### What Backend Sees vs What It Never Sees")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Backend RECEIVES (Encrypted/Hashed Only):**
                - ZK Commitment Hash: `261f251103c13978...`
                - Proof Hash: `3a8d9a6dea660b6b...`
                - Encrypted Location: `917520...` (encrypted blob)
                - Encrypted Time: `1152921504604880868...` (encrypted blob)
                - Encrypted Device: `1152921504606126078...` (encrypted blob)
                - Encrypted Pincode: `1152921504605208553...` (encrypted blob)
                """)
            
            with col2:
                st.markdown("""
                **Backend NEVER SEES:**
                - Plaintext Name: "Vatsal"
                - Plaintext UID: "123456789012"
                - Plaintext Pincode: "575025"
                - Plaintext Location Risk: 2
                - Plaintext Time Risk: 1
                - Plaintext Device Trust: 8
                - Plaintext Pincode Risk: 1
                """)
            
            # Show recent request details
            recent_requests = {}
            for log in reversed(logs):
                if log['operation'] == 'zk_verification_start':
                    timestamp = log['timestamp']
                    if timestamp not in recent_requests:
                        recent_requests[timestamp] = {'zk_start': log}
                elif log['operation'] == 'ml_inference_start':
                    timestamp = log['timestamp']
                    if timestamp in recent_requests:
                        recent_requests[timestamp]['ml_start'] = log
                elif log['operation'] == 'ml_inference_complete':
                    timestamp = log['timestamp']
                    if timestamp in recent_requests:
                        recent_requests[timestamp]['ml_complete'] = log
            
            for timestamp, req_data in list(recent_requests.items())[:5]:
                with st.expander(f"Request at {timestamp}"):
                    if 'zk_start' in req_data:
                        st.markdown("**ZK Verification:**")
                        st.json(req_data['zk_start']['details'])
                    if 'ml_start' in req_data:
                        st.markdown("**ML Inference Start:**")
                        st.json(req_data['ml_start']['details'])
                    if 'ml_complete' in req_data:
                        st.markdown("**ML Inference Complete:**")
                        st.json(req_data['ml_complete']['details'])
        else:
            st.warning("Could not fetch logs")
    except Exception as e:
        st.error(f"Error: {e}")
    
    # Show input data from logs
    st.markdown("### Input Data Received (Encrypted Only)")
    try:
        logs_resp = requests.get(f"{CLOUD_URL}/logs", params={"limit": 50, "operation": "ml_inference_start"}, timeout=5)
        if logs_resp.status_code == 200:
            logs = logs_resp.json().get('logs', [])
            if logs:
                st.markdown("**Recent Encrypted Inputs (Backend Only Sees Encrypted Blobs):**")
                for log in reversed(logs[-5:]):
                    details = log.get('details', {})
                    encrypted_inputs = details.get('encrypted_inputs', {})
                    st.markdown(f"**Request at {log.get('timestamp', 'unknown')}:**")
                    st.code(f"""
Location (encrypted): {encrypted_inputs.get('location_prefix', 'N/A')}
Time (encrypted):     {encrypted_inputs.get('time_prefix', 'N/A')}
Device (encrypted):   {encrypted_inputs.get('device_prefix', 'N/A')}
Pincode (encrypted):  {encrypted_inputs.get('pincode_prefix', 'N/A')}
                    """)
                    st.markdown("*Note: Backend cannot decrypt these values - they remain encrypted throughout processing*")
            else:
                st.info("No ML inference logs yet")
        
        # Show HE operations
        he_logs_resp = requests.get(f"{CLOUD_URL}/logs", params={"limit": 20, "operation": "he_operation"}, timeout=5)
        if he_logs_resp.status_code == 200:
            he_logs = he_logs_resp.json().get('logs', [])
            if he_logs:
                st.markdown("**Homomorphic Operations (On Encrypted Data):**")
                for log in reversed(he_logs[-10:]):
                    details = log.get('details', {})
                    st.text(f"Operation: {details.get('operation', 'unknown')} | Feature: {details.get('feature', 'unknown')} | Weight: {details.get('weight', 'N/A')} | Result (encrypted): {details.get('result_prefix', 'N/A')}...")
    except Exception as e:
        st.warning(f"Could not fetch input data: {e}")

with tab4:
    st.header("ML Model Processing")
    
    st.markdown("### Privacy-Preserving Machine Learning Inference")
    
    # ML Model Info
    try:
        health_resp = requests.get(f"{CLOUD_URL}/health", timeout=5)
        if health_resp.status_code == 200:
            health_data = health_resp.json()
            if health_data.get('ml_model_loaded'):
                st.success("ML Model Loaded Successfully")
            else:
                st.warning("ML Model Not Loaded - Using Default Weights")
    except:
        st.warning("Could not check ML model status")
    
    # ML Processing Flow
    st.markdown("#### ML Model Computation Flow")
    st.markdown("""
    **Model Type:** Logistic Regression (Linear Model)
    
    **Formula:** Risk Score = (w1 × Enc(location) + w2 × Enc(time) + w3 × Enc(device) + w4 × Enc(pincode) + bias)
    
    **Steps:**
    1. Receive encrypted risk factors from client
    2. Multiply each encrypted value by corresponding weight (homomorphic multiplication)
    3. Add all weighted terms together (homomorphic addition)
    4. Add bias term
    5. Return encrypted result to client
    
    **Privacy:** Cloud never sees plaintext values, only encrypted blobs
    """)
    
    # ML Processing Logs
    st.markdown("#### Recent ML Inference Operations")
    
    try:
        logs_resp = requests.get(f"{CLOUD_URL}/logs", params={"limit": 100}, timeout=5)
        if logs_resp.status_code == 200:
            logs = logs_resp.json().get('logs', [])
            ml_logs = [log for log in logs if 'ml' in log['operation'].lower() or 'he_operation' in log['operation']]
            
            if ml_logs:
                # Group by inference session
                sessions = {}
                for log in reversed(ml_logs):
                    if log['operation'] == 'ml_inference_start':
                        session_id = log['timestamp']
                        sessions[session_id] = {'start': log, 'operations': []}
                    elif log['operation'] == 'he_operation':
                        # Find most recent session
                        if sessions:
                            latest_session = list(sessions.keys())[-1]
                            sessions[latest_session]['operations'].append(log)
                    elif log['operation'] == 'ml_inference_complete':
                        session_id = log['timestamp']
                        if session_id in sessions:
                            sessions[session_id]['complete'] = log
                
                for session_id, session_data in list(sessions.items())[:5]:
                    with st.expander(f"ML Inference Session: {session_id}"):
                        if 'start' in session_data:
                            st.markdown("**Model Weights Used:**")
                            weights = session_data['start']['details'].get('model_weights', {})
                            st.json(weights)
                        
                        if 'operations' in session_data and session_data['operations']:
                            st.markdown("**Homomorphic Operations:**")
                            for op in session_data['operations']:
                                details = op['details']
                                st.text(f"{details.get('operation', 'unknown')} - Feature: {details.get('feature', 'unknown')}, Weight: {details.get('weight', 'N/A')}")
                        
                        if 'complete' in session_data:
                            st.markdown("**Result:**")
                            st.json(session_data['complete']['details'])
            else:
                st.info("No ML processing logs yet. Make some inference requests!")
        else:
            st.warning("Could not fetch ML logs")
    except Exception as e:
        st.error(f"Error fetching ML logs: {e}")
    
    # Model Architecture
    st.markdown("#### Model Architecture")
    st.markdown("""
    **Current Model: Logistic Regression**
    
    **Input Features (Encrypted):**
    - Location Risk (0-10)
    - Time Risk (0-10)
    - Device Trust (0-10)
    - Pincode Risk (0-10)
    
    **Homomorphic Operations:**
    1. Encrypted scalar multiplication for each feature
    2. Homomorphic addition of weighted terms
    3. Bias addition
    4. Result remains encrypted until client decryption
    
    **Output:** Encrypted Anomaly Score
    """)

with tab5:
    st.header("Performance Benchmarks")
    
    st.markdown("### Cryptographic Operation Benchmarks")
    
    # Benchmark endpoints
    endpoints = {
        "Latency": "/benchmarks/latency",
        "Depth": "/benchmarks/depth",
        "Threading": "/benchmarks/threading",
        "Summary": "/benchmarks/summary"
    }
    
    selected_benchmark = st.selectbox("Select Benchmark Type", list(endpoints.keys()))
    
    try:
        bench_resp = requests.get(f"{BENCHMARK_URL}{endpoints[selected_benchmark]}", timeout=5)
        
        if bench_resp.status_code == 200:
            bench_data = bench_resp.json()
            
            if bench_data.get('status') == 'success':
                data = bench_data.get('data', [])
                
                if data:
                    df = pd.DataFrame(data)
                    
                    # Display statistics if available
                    if 'statistics' in bench_data:
                        st.subheader("Statistics")
                        stats_df = pd.DataFrame(bench_data['statistics']).T
                        st.dataframe(stats_df)
                    
                    # Display data
                    st.subheader("Raw Data")
                    st.dataframe(df)
                    
                    # Visualizations
                    if selected_benchmark == "Latency":
                        st.subheader("Latency Visualization")
                        if 'Operation' in df.columns and 'Latency (ms)' in df.columns:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            df_grouped = df.groupby('Operation')['Latency (ms)'].mean().sort_values()
                            df_grouped.plot(kind='barh', ax=ax, color='steelblue')
                            ax.set_xlabel('Average Latency (ms)')
                            ax.set_title('Operation Latency Comparison')
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    elif selected_benchmark == "Depth":
                        st.subheader("Depth Impact Visualization")
                        if 'Depth' in df.columns and 'Time (ms)' in df.columns:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            for type_val in df['Type'].unique() if 'Type' in df.columns else []:
                                type_data = df[df['Type'] == type_val]
                                ax.plot(type_data['Depth'], type_data['Time (ms)'], 
                                       marker='o', label=type_val)
                            ax.set_xlabel('Computation Depth')
                            ax.set_ylabel('Time (ms)')
                            ax.set_title('Computation Depth Impact')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    elif selected_benchmark == "Threading":
                        st.subheader("Threading Performance")
                        if 'Threads' in df.columns and 'Throughput (Enc/s)' in df.columns:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(df['Threads'], df['Throughput (Enc/s)'], 
                                   marker='o', color='green', linewidth=2)
                            ax.set_xlabel('Number of Threads')
                            ax.set_ylabel('Throughput (Encryptions/sec)')
                            ax.set_title('Parallel Processing Performance')
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                else:
                    st.info("No benchmark data available")
            else:
                st.warning(f"Benchmark service returned: {bench_data.get('message', 'Unknown error')}")
        else:
            st.error(f"Failed to fetch benchmarks: {bench_resp.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to benchmark service. Is it running?")
    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("Admin Panel - Versatile Verification Cloud Framework")
