-- Add retention analysis columns to resumes table
-- This includes institution tier information and retention scoring

-- Add institution tier columns
ALTER TABLE resumes
ADD COLUMN IF NOT EXISTS ug_tier INTEGER,
ADD COLUMN IF NOT EXISTS pg_tier INTEGER;

-- Add retention analysis columns
ALTER TABLE resumes
ADD COLUMN IF NOT EXISTS retention_score NUMERIC(5,2),
ADD COLUMN IF NOT EXISTS retention_risk VARCHAR(20) CHECK (retention_risk IN ('Low', 'Medium', 'High')),
ADD COLUMN IF NOT EXISTS retention_analysis JSONB;

-- Add constraints for tier values (1-3 only, or NULL if empty)
-- Use DO block to check if constraints exist before adding them
DO $$
BEGIN
    -- Drop old constraints if they exist (in case they allowed 1-5)
    ALTER TABLE resumes DROP CONSTRAINT IF EXISTS chk_ug_tier_range;
    ALTER TABLE resumes DROP CONSTRAINT IF EXISTS chk_pg_tier_range;
    
    -- Add constraint for ug_tier if it doesn't exist (only 1-3 or NULL)
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'chk_ug_tier_range' 
        AND conrelid = 'resumes'::regclass
    ) THEN
        ALTER TABLE resumes
        ADD CONSTRAINT chk_ug_tier_range CHECK (ug_tier IS NULL OR (ug_tier >= 1 AND ug_tier <= 3));
    END IF;

    -- Add constraint for pg_tier if it doesn't exist (only 1-3 or NULL)
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'chk_pg_tier_range' 
        AND conrelid = 'resumes'::regclass
    ) THEN
        ALTER TABLE resumes
        ADD CONSTRAINT chk_pg_tier_range CHECK (pg_tier IS NULL OR (pg_tier >= 1 AND pg_tier <= 3));
    END IF;
END $$;

-- Add index for retention score queries
CREATE INDEX IF NOT EXISTS idx_resumes_retention_score ON resumes(retention_score);
CREATE INDEX IF NOT EXISTS idx_resumes_retention_risk ON resumes(retention_risk);

-- Add comment for documentation
COMMENT ON COLUMN resumes.ug_tier IS 'Undergraduate institution tier (1=Top-tier, 2=Good, 3=Other, NULL=Empty/Not provided)';
COMMENT ON COLUMN resumes.pg_tier IS 'Postgraduate institution tier (1=Top-tier, 2=Good, 3=Other, NULL=Empty/Not provided)';
COMMENT ON COLUMN resumes.retention_score IS 'Overall retention risk score (0-100, higher is better)';
COMMENT ON COLUMN resumes.retention_risk IS 'Retention risk level: Low, Medium, or High';
COMMENT ON COLUMN resumes.retention_analysis IS 'JSON object containing detailed retention analysis including component scores, risk flags, and recommendations';

