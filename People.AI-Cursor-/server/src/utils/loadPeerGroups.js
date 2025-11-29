// Utility to load peer groups from JSON file
// The .pkl file should be converted to JSON using a Python script
// This JSON file will be loaded at server startup

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { retentionScorer } from '../services/retentionService.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Load peer groups from JSON file
 * Expected file structure:
 * {
 *   "peer_groups": {
 *     "Experienced_Professionals": [...],
 *     "Junior_Professionals": [...],
 *     ...
 *   }
 * }
 */
export function loadPeerGroupsFromFile(filePath = null) {
    try {
        // Default path: server/data/peer_groups.json
        const defaultPath = path.join(__dirname, '../../data/peer_groups.json');
        const peerGroupsPath = filePath || defaultPath;

        // Check if file exists
        if (!fs.existsSync(peerGroupsPath)) {
            console.warn(`⚠ Peer groups file not found at: ${peerGroupsPath}`);
            console.warn('⚠ Retention scoring will work without peer comparison');
            console.warn('⚠ To enable peer comparison, convert your .pkl file to JSON and place it at:');
            console.warn(`   ${peerGroupsPath}`);
            return false;
        }

        // Read and parse JSON file
        const fileContent = fs.readFileSync(peerGroupsPath, 'utf8');
        const data = JSON.parse(fileContent);

        // Load peer groups into retention scorer
        if (data.peer_groups && typeof data.peer_groups === 'object') {
            retentionScorer.loadPeerGroups(data.peer_groups);
            console.log(`✓ Loaded ${Object.keys(data.peer_groups).length} peer groups from ${peerGroupsPath}`);
            return true;
        } else {
            console.warn('⚠ Invalid peer groups file format. Expected { "peer_groups": {...} }');
            return false;
        }
    } catch (error) {
        console.error('Error loading peer groups:', error);
        console.warn('⚠ Retention scoring will work without peer comparison');
        return false;
    }
}

/**
 * Create a Python script template for converting .pkl to JSON
 * This will be saved as a helper script
 */
export function createPklToJsonScript() {
    const scriptContent = `# Convert retention_scorer_with_peers.pkl to JSON for Node.js
# Run this script: python convert_pkl_to_json.py

import pickle
import json
from pathlib import Path

# Load the .pkl file
pkl_path = 'retention_scorer_with_peers.pkl'
json_path = 'peer_groups.json'

print(f"Loading {pkl_path}...")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# Extract peer groups
peer_groups = data.get('peer_groups', {})

# Convert to JSON-serializable format
# Ensure all data types are JSON-compatible
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)

json_data = {
    'peer_groups': make_json_serializable(peer_groups)
}

# Save to JSON file
print(f"Saving to {json_path}...")
with open(json_path, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"✓ Successfully converted {pkl_path} to {json_path}")
print(f"✓ Found {len(peer_groups)} peer groups")
print(f"\\nNext steps:")
print(f"1. Copy {json_path} to: server/data/peer_groups.json")
print(f"2. Restart your Node.js server")
`;

    const scriptPath = path.join(__dirname, '../../convert_pkl_to_json.py');
    fs.writeFileSync(scriptPath, scriptContent);
    console.log(`✓ Created conversion script at: ${scriptPath}`);
    console.log('  Run: python convert_pkl_to_json.py');
}


