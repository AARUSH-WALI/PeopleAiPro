# Retention Scoring Integration Guide

This guide explains how to integrate the new retention scoring system with peer comparison into your application.

## 📋 Overview

The retention scoring system has been fully integrated into your codebase. It calculates retention risk based on:
- **Job Stability** (25%): Average tenure and job changes
- **Personality Fit** (35%): Big5 personality scores (Conscientiousness, Agreeableness, Neuroticism)
- **Professional Engagement** (20%): Workshops, trainings, papers, patents, achievements
- **Fitment Factor** (15%): Overall fitment score
- **Academic Tier** (5%): UG and PG institution tiers
- **Peer Comparison** (20% when available): Comparison with historical candidates

## 🚀 Quick Start

### Step 1: Convert .pkl File to JSON

1. Place your `retention_scorer_with_peers.pkl` file in the project root directory

2. Run the conversion script:
   ```bash
   python convert_pkl_to_json.py
   ```

3. Copy the generated `peer_groups.json` to:
   ```
   server/data/peer_groups.json
   ```

### Step 2: Run Database Migration

Run the new migration to add peer comparison columns:

```sql
-- Run this in your Supabase SQL editor or via migration script
-- File: server/migrations/012_add_retention_peer_comparison_columns.sql
```

Or run via your migration script:
```bash
node server/src/scripts/runMigration.js 012_add_retention_peer_comparison_columns.sql
```

### Step 3: Restart Server

Restart your Node.js server. You should see in the logs:
```
Loading peer groups for retention scoring...
✓ Loaded X peer groups from server/data/peer_groups.json
```

## 📁 File Locations

### Backend Files
- **Retention Service**: `server/src/services/retentionService.js`
- **Controller Integration**: `server/src/controllers/resumeController.js` (updated `updatePersonalityTestResults`)
- **Peer Groups Loader**: `server/src/utils/loadPeerGroups.js`
- **Routes**: `server/src/routes/retention.js` (backfill endpoint)
- **Migration**: `server/migrations/012_add_retention_peer_comparison_columns.sql`

### Frontend Files
- **UI Component**: `client/src/components/users/RetentionAnalysis.tsx` (updated)
- **Profile Page**: `client/src/pages/CandidateProfile.tsx` (retention analysis re-enabled)

### Data Files
- **Peer Groups JSON**: `server/data/peer_groups.json` (you need to create this)
- **Conversion Script**: `convert_pkl_to_json.py` (in project root)

## 🔄 How It Works

### Automatic Calculation Flow

1. **Personality Test Completion**:
   - User completes Big5 personality test
   - Frontend calls: `PATCH /api/resumes/personality-test`

2. **Backend Processing**:
   - Updates personality scores in database
   - Calculates fitment score
   - **Calculates retention risk** (NEW)
   - Determines peer group automatically
   - Calculates peer comparison score (if peer groups loaded)

3. **Database Storage**:
   - Saves `retention_score` (0-100)
   - Saves `retention_risk` (Low/Medium/High)
   - Saves `retention_analysis` (JSONB with full details)
   - Saves component scores in individual columns:
     - `retention_stability_score`
     - `retention_personality_score`
     - `retention_engagement_score`
     - `retention_fitment_factor`
     - `retention_academic_tier_score`
     - `retention_peer_comp_score`

4. **UI Display**:
   - Retention analysis appears in candidate profile
   - Shows gauge, risk level, component scores, flags, and insights

## 📊 Database Schema

### New Columns Added

| Column Name | Type | Description |
|------------|------|-------------|
| `retention_score` | NUMERIC(5,2) | Overall retention score (0-100) |
| `retention_risk` | VARCHAR(20) | Risk level: Low, Medium, High |
| `retention_analysis` | JSONB | Full detailed analysis |
| `retention_stability_score` | NUMERIC(5,2) | Job stability component (0-100) |
| `retention_personality_score` | NUMERIC(5,2) | Personality fit component (0-100) |
| `retention_engagement_score` | NUMERIC(5,2) | Professional engagement component (0-100) |
| `retention_fitment_factor` | NUMERIC(5,2) | Fitment factor component (0-100) |
| `retention_academic_tier_score` | NUMERIC(5,2) | Academic tier component (0-100) |
| `retention_peer_comp_score` | NUMERIC(5,2) | Peer comparison component (0-100, nullable) |

## 🔧 Configuration

### Peer Groups

The system automatically determines peer groups based on:
1. **Category-based**: `Experienced_Professionals`, `Intermediate_Professionals`, `Fresher_Professionals`
2. **Experience-based**: `Junior_Professionals` (0-3 years), `Mid_Level_Professionals` (4-8 years), `Senior_Professionals` (8+ years)
3. **All Historical**: `All_Historical_Data`

If a matching peer group is found, peer comparison is included (20% weight). Otherwise, base calculation is used.

### Without Peer Groups

If `peer_groups.json` is not found or empty:
- Retention scoring still works
- Uses base calculation (without peer comparison)
- Logs a warning but continues normally

## 📝 API Endpoints

### Update Personality Test (with Retention)
```
PATCH /api/resumes/personality-test
Body: {
  email: "candidate@example.com",
  extraversion: 25,
  agreeableness: 30,
  conscientiousness: 35,
  neuroticism: 15,
  openness: 28
}
```

**Response includes retention data:**
```json
{
  "success": true,
  "data": {
    "retention_score": 72.5,
    "retention_risk": "Low",
    "retention_analysis": { ... },
    "retention_stability_score": 80.0,
    "retention_personality_score": 75.0,
    ...
  }
}
```

### Backfill Retention (for existing candidates)
```
POST /api/retention/backfill
```

Recalculates retention scores for all candidates with personality test data.

## 🎨 UI Components

### RetentionAnalysis Component

Displays:
- **Retention Score Gauge**: Visual gauge showing score (0-100)
- **Risk Level Badge**: Color-coded risk level (Low/Medium/High)
- **Component Scores**: Breakdown of all components
- **Risk Flags**: List of identified risk factors
- **Insights**: Actionable recommendations
- **Peer Group Info**: Shows which peer group was used (if applicable)

### Location
- **Candidate Profile Page**: `/users/:id` or candidate detail view
- Automatically appears when `retention_score` is available

## 🧪 Testing

### Test Retention Calculation

1. Complete a personality test for a candidate
2. Check the database for retention fields:
   ```sql
   SELECT 
     retention_score, 
     retention_risk, 
     retention_stability_score,
     retention_personality_score,
     retention_peer_comp_score
   FROM resumes 
   WHERE email = 'test@example.com';
   ```

3. View candidate profile - retention analysis should appear

### Test Peer Comparison

1. Ensure `server/data/peer_groups.json` exists
2. Restart server - check logs for peer groups loaded
3. Complete personality test - check if `retention_peer_comp_score` is populated

## 🐛 Troubleshooting

### Issue: Peer groups not loading

**Symptoms**: Server logs show "Peer groups file not found"

**Solutions**:
1. Check file exists: `server/data/peer_groups.json`
2. Verify JSON format: Should have `{"peer_groups": {...}}` structure
3. Check file permissions

### Issue: Peer comparison score is null

**Possible causes**:
1. No matching peer group found for candidate
2. Peer groups file not loaded
3. Peer group exists but has no candidates

**Solution**: Check server logs and verify peer groups JSON structure

### Issue: Retention scores seem incorrect

**Check**:
1. Verify all required fields are populated in database:
   - `longevity_years`
   - `number_of_unique_designations` or `number_of_jobs`
   - `workshops_count` or `workshops`
   - `trainings_count` or `trainings`
   - Big5 personality scores

2. Check component scores in `retention_analysis` JSONB field

### Issue: Migration fails

**Solution**: Run migration manually in Supabase SQL editor:
```sql
-- Copy contents of server/migrations/012_add_retention_peer_comparison_columns.sql
```

## 📚 Additional Resources

- **Retention Service Code**: `server/src/services/retentionService.js`
- **Database Migrations**: `server/migrations/`
- **Peer Groups Data**: `server/data/README.md`

## ✅ Checklist

Before going to production:

- [ ] Convert `.pkl` file to JSON
- [ ] Place `peer_groups.json` in `server/data/`
- [ ] Run database migration `012_add_retention_peer_comparison_columns.sql`
- [ ] Restart server and verify peer groups load
- [ ] Test retention calculation with a new candidate
- [ ] Verify retention data appears in candidate profile
- [ ] Test backfill endpoint for existing candidates
- [ ] Check database columns are populated correctly

## 🎯 Next Steps

1. **Convert your .pkl file**: Run `python convert_pkl_to_json.py`
2. **Place JSON file**: Copy to `server/data/peer_groups.json`
3. **Run migration**: Execute `012_add_retention_peer_comparison_columns.sql`
4. **Test**: Complete a personality test and verify retention analysis appears

---

**Questions?** Check the code comments in `server/src/services/retentionService.js` for detailed implementation notes.


