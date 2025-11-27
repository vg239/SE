"""
Benchmarking Service for HE Operations
Integrates with existing experimental data and provides API endpoints for benchmarking.
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os
import json
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Paths to benchmark data
BENCHMARK_DATA_DIR = "/app/benchmarks"
EXP1_PATH = os.path.join(BENCHMARK_DATA_DIR, "exp1_latency_results.csv")
EXP2_PATH = os.path.join(BENCHMARK_DATA_DIR, "exp2_depth_results.csv")
EXP3_PATH = os.path.join(BENCHMARK_DATA_DIR, "exp3_threading_results.csv")

def load_benchmark_data():
    """Load benchmark data from CSV files"""
    data = {}
    
    try:
        if os.path.exists(EXP1_PATH):
            df1 = pd.read_csv(EXP1_PATH)
            data['latency'] = df1.to_dict('records')
            # Calculate statistics
            stats = df1.groupby('Operation')['Latency (ms)'].agg(['mean', 'std', 'min', 'max']).to_dict('index')
            data['latency_stats'] = stats
    except Exception as e:
        print(f"Error loading exp1: {e}")
    
    try:
        if os.path.exists(EXP2_PATH):
            df2 = pd.read_csv(EXP2_PATH)
            data['depth'] = df2.to_dict('records')
    except Exception as e:
        print(f"Error loading exp2: {e}")
    
    try:
        if os.path.exists(EXP3_PATH):
            df3 = pd.read_csv(EXP3_PATH)
            data['threading'] = df3.to_dict('records')
    except Exception as e:
        print(f"Error loading exp3: {e}")
    
    return data

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Benchmarking Service"
    })

@app.route('/benchmarks/latency', methods=['GET'])
def get_latency_benchmarks():
    """Get latency benchmarks for HE operations"""
    data = load_benchmark_data()
    
    if 'latency' not in data:
        return jsonify({
            "status": "error",
            "message": "Latency benchmark data not found"
        }), 404
    
    return jsonify({
        "status": "success",
        "data": data['latency'],
        "statistics": data.get('latency_stats', {}),
        "summary": {
            "total_operations": len(data['latency']),
            "operations": list(data.get('latency_stats', {}).keys())
        }
    })

@app.route('/benchmarks/depth', methods=['GET'])
def get_depth_benchmarks():
    """Get computation depth benchmarks"""
    data = load_benchmark_data()
    
    if 'depth' not in data:
        return jsonify({
            "status": "error",
            "message": "Depth benchmark data not found"
        }), 404
    
    return jsonify({
        "status": "success",
        "data": data['depth']
    })

@app.route('/benchmarks/threading', methods=['GET'])
def get_threading_benchmarks():
    """Get threading/parallelism benchmarks"""
    data = load_benchmark_data()
    
    if 'threading' not in data:
        return jsonify({
            "status": "error",
            "message": "Threading benchmark data not found"
        }), 404
    
    return jsonify({
        "status": "success",
        "data": data['threading']
    })

@app.route('/benchmarks/summary', methods=['GET'])
def get_benchmark_summary():
    """Get summary of all benchmarks"""
    data = load_benchmark_data()
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "latency": {
            "available": 'latency' in data,
            "operations": len(data.get('latency', [])),
            "statistics": data.get('latency_stats', {})
        },
        "depth": {
            "available": 'depth' in data,
            "records": len(data.get('depth', []))
        },
        "threading": {
            "available": 'threading' in data,
            "records": len(data.get('threading', []))
        }
    }
    
    return jsonify({
        "status": "success",
        "summary": summary
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)


