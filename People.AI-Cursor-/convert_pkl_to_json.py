#!/usr/bin/env python3
"""
Convert retention_scorer_with_peers.pkl to JSON for Node.js backend

This script extracts peer groups from the Python pickle file and converts
them to a JSON format that can be loaded by the Node.js retention service.

Usage:
    python convert_pkl_to_json.py

Requirements:
    - retention_scorer_with_peers.pkl file in the same directory
"""

import pickle
import json
from pathlib import Path

def make_json_serializable(obj):
    """Convert Python objects to JSON-serializable format"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    elif isinstance(obj, (complex, bytes)):
        return str(obj)
    else:
        # For any other type, convert to string
        try:
            return str(obj)
        except:
            return None

def convert_pkl_to_json(pkl_path='retention_scorer_with_peers.pkl', json_path='peer_groups.json'):
    """Convert .pkl file to JSON format"""
    
    # Check if .pkl file exists
    if not Path(pkl_path).exists():
        print(f"❌ ERROR: File '{pkl_path}' not found!")
        print(f"\nPlease make sure:")
        print(f"1. The file '{pkl_path}' is in the current directory")
        print(f"2. OR update the pkl_path variable with the correct path")
        return False
    
    print(f"📂 Loading {pkl_path}...")
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ ERROR: Failed to load pickle file: {e}")
        return False
    
    # Extract peer groups
    peer_groups = data.get('peer_groups', {})
    
    if not peer_groups:
        print("⚠ WARNING: No peer groups found in the pickle file")
        print("The retention scorer will work without peer comparison")
        peer_groups = {}
    
    print(f"✓ Found {len(peer_groups)} peer groups")
    
    # Count total candidates in all peer groups
    total_candidates = sum(len(candidates) for candidates in peer_groups.values())
    print(f"✓ Total candidates across all peer groups: {total_candidates}")
    
    # Convert to JSON-serializable format
    print("🔄 Converting to JSON format...")
    json_data = {
        'peer_groups': make_json_serializable(peer_groups),
        'metadata': {
            'total_peer_groups': len(peer_groups),
            'total_candidates': total_candidates,
            'peer_group_names': list(peer_groups.keys())
        }
    }
    
    # Save to JSON file
    print(f"💾 Saving to {json_path}...")
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Successfully converted to {json_path}")
    except Exception as e:
        print(f"❌ ERROR: Failed to save JSON file: {e}")
        return False
    
    # Show summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Peer Groups: {len(peer_groups)}")
    for group_name, candidates in peer_groups.items():
        print(f"  - {group_name}: {len(candidates)} candidates")
    print(f"\nTotal Candidates: {total_candidates}")
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print(f"1. Copy {json_path} to: server/data/peer_groups.json")
    print(f"2. Restart your Node.js server")
    print(f"3. Check server logs for: '✓ Loaded {len(peer_groups)} peer groups'")
    print("\n✅ Conversion complete!")
    
    return True

if __name__ == "__main__":
    import sys
    
    # Allow custom paths via command line arguments
    pkl_path = sys.argv[1] if len(sys.argv) > 1 else 'retention_scorer_with_peers.pkl'
    json_path = sys.argv[2] if len(sys.argv) > 2 else 'peer_groups.json'
    
    print("="*60)
    print("RETENTION SCORER - PKL TO JSON CONVERTER")
    print("="*60)
    print()
    
    success = convert_pkl_to_json(pkl_path, json_path)
    
    if not success:
        sys.exit(1)


