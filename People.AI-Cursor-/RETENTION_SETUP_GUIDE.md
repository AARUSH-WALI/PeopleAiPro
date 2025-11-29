# Retention Analysis Setup Guide

## ✅ Completed Steps

1. ✅ Database migration created and run (`006_add_retention_analysis.sql`)
2. ✅ Retention scorer service implemented
3. ✅ Backend integration complete
4. ✅ UI components created and integrated

## 🚀 Next Steps

### 1. Backfill Existing Candidates

Run the backfill script to calculate retention for existing candidates:

**Using API:**
```bash
POST http://localhost:5000/api/retention/backfill
```

**Response:**
```json
{
  "success": true,
  "message": "Retention analysis backfill completed",
  "processed": 10,
  "errors": 0,
  "total": 10
}
```

### 2. Test with New Candidates

When a candidate completes the personality test, retention analysis is automatically calculated and stored.

### 3. View Retention Analysis

- Navigate to any candidate profile page
- Retention analysis appears below the personality assessment
- Shows:
  - Overall retention score (0-100)
  - Risk level (Low/Medium/High)
  - Component breakdown (Job Stability, Personality Fit, Engagement, Fitment)
  - Risk flags
  - Key recommendations

### 4. (Optional) Set Institution Tiers

Currently, `ug_tier` and `pg_tier` default to 5 if not set. To improve accuracy:

1. **Manual Update:** Update tiers directly in the database
2. **Automatic Detection:** Create a mapping service to determine tiers based on institution names
3. **Import Script:** Create a script to import tier data from an external source

Example SQL to update tiers:
```sql
UPDATE resumes 
SET ug_tier = 1, pg_tier = 1 
WHERE ug_institute = 'IIT' OR pg_institute = 'IIT';
```

## 📊 How It Works

1. **Personality Test Completion** → `updatePersonalityTestResults` endpoint called
2. **Fitment Calculation** → Overall fitment score calculated
3. **Retention Calculation** → Retention risk calculated using:
   - Job stability (30%) - includes institution tiers (5%)
   - Personality fit (35%)
   - Professional engagement (20%)
   - Fitment factor (15%)
4. **Storage** → Results stored in `retention_score`, `retention_risk`, and `retention_analysis` columns
5. **Display** → UI shows comprehensive retention analysis

## 🔧 Troubleshooting

### Retention data not showing?
- Ensure the candidate has completed the personality test
- Run the backfill script for existing candidates
- Check server logs for any errors

### Scores seem incorrect?
- Verify institution tiers are set correctly
- Check that all candidate data (longevity, jobs, etc.) is accurate
- Review the retention scorer service logic if needed

### Need to recalculate?
- Simply trigger the personality test update again
- Or run the backfill script to recalculate all candidates

## 📝 Files Created/Modified

- `server/src/services/retentionService.js` - Retention calculation logic
- `server/src/scripts/backfillRetention.js` - Backfill script
- `server/src/routes/retention.js` - Backfill API endpoint
- `server/src/controllers/resumeController.js` - Integration with personality test
- `server/migrations/006_add_retention_analysis.sql` - Database migration
- `client/src/components/users/RetentionAnalysis.tsx` - UI component
- `client/src/pages/CandidateProfile.tsx` - Updated to show retention
- `client/src/components/dashboard/CandidateDetailsDialog.tsx` - Updated to show retention



