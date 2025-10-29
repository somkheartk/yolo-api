#!/usr/bin/env python3
"""
Simple test script for YOLO API
"""
import requests
import json
import sys
from pathlib import Path

def test_health_endpoint(base_url):
    """Test the health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        data = response.json()
        print(f"✓ Health check passed: {json.dumps(data, indent=2)}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_root_endpoint(base_url):
    """Test the root endpoint"""
    print("\nTesting / endpoint...")
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()
        print(f"✓ Root endpoint passed: {json.dumps(data, indent=2)}")
        return True
    except Exception as e:
        print(f"✗ Root endpoint failed: {e}")
        return False

def test_model_info_endpoint(base_url):
    """Test the model info endpoint"""
    print("\nTesting /model-info endpoint...")
    try:
        response = requests.get(f"{base_url}/model-info")
        response.raise_for_status()
        data = response.json()
        print(f"✓ Model info passed: {json.dumps(data, indent=2)}")
        return True
    except Exception as e:
        print(f"✗ Model info failed: {e}")
        return False

def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print(f"Testing YOLO API at {base_url}")
    print("=" * 50)
    
    results = []
    results.append(test_root_endpoint(base_url))
    results.append(test_health_endpoint(base_url))
    results.append(test_model_info_endpoint(base_url))
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
