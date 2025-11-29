# Retention Peer Groups Data

This directory contains the peer groups data for retention scoring with peer comparison.

## Setup Instructions

### Step 1: Convert .pkl to JSON

1. Place your `retention_scorer_with_peers.pkl` file in the project root (or any accessible location)

2. Run the Python conversion script:
   ```bash
   python convert_pkl_to_json.py
   ```

   Or manually convert using this Python script:

   ```python
   import pickle
   import json
   
   # Load the .pkl file
   with open('retention_scorer_with_peers.pkl', 'rb') as f:
       data = pickle.load(f)
   
   # Extract peer groups
   peer_groups = data.get('peer_groups', {})
   
   # Convert to JSON-serializable format
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
   with open('peer_groups.json', 'w') as f:
       json.dump(json_data, f, indent=2)
   
   print(f"✓ Converted to peer_groups.json")
   print(f"✓ Found {len(peer_groups)} peer groups")
   ```

### Step 2: Place JSON File

Copy the generated `peer_groups.json` file to this directory:
```
server/data/peer_groups.json
```

### Step 3: Verify

The server will automatically load peer groups on startup. Check the server logs for:
```
✓ Loaded X peer groups from server/data/peer_groups.json
```

## File Structure

The `peer_groups.json` file should have this structure:

```json
{
  "peer_groups": {
    "Experienced_Professionals": [
      {
        "longevity_years": 5.0,
        "number_of_unique_designations": 2,
        "workshops": 3,
        "trainings": 4,
        "fitment_score": 75.0,
        "retention_score": 72.5,
        "component_scores": {
          "stability": 80.0,
          "personality": 75.0,
          "engagement": 70.0,
          "fitment_factor": 82.0,
          "tier_score": 60.0
        }
      },
      ...
    ],
    "Junior_Professionals": [...],
    "Mid_Level_Professionals": [...],
    "Senior_Professionals": [...],
    "All_Historical_Data": [...]
  }
}
```

## Notes

- If the peer groups file is not found, retention scoring will still work but **without peer comparison**
- Peer comparison adds 20% weight to the retention score when available
- Without peer groups, the retention score uses the base calculation (stability, personality, engagement, fitment, tier)

## Troubleshooting

**Issue**: "Peer groups file not found"
- **Solution**: Make sure `peer_groups.json` is in `server/data/` directory

**Issue**: "Invalid peer groups file format"
- **Solution**: Check that the JSON has `{"peer_groups": {...}}` structure

**Issue**: Peer comparison not working
- **Solution**: Verify that peer groups contain candidates with `retention_score` and `component_scores` fields


