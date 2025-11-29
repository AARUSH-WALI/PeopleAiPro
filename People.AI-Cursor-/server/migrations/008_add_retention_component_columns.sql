-- Add individual component score columns for retention analysis
-- This makes it easier to query and display component breakdowns
-- without needing to parse the JSONB retention_analysis field

ALTER TABLE resumes
ADD COLUMN IF NOT EXISTS retention_stability_score NUMERIC(5,2),
ADD COLUMN IF NOT EXISTS retention_personality_score NUMERIC(5,2),
ADD COLUMN IF NOT EXISTS retention_engagement_score NUMERIC(5,2),
ADD COLUMN IF NOT EXISTS retention_fitment_factor NUMERIC(5,2),
ADD COLUMN IF NOT EXISTS retention_institution_quality NUMERIC(5,2);

-- Add comments for documentation
COMMENT ON COLUMN resumes.retention_stability_score IS 'Job stability component score (0-100) for retention analysis';
COMMENT ON COLUMN resumes.retention_personality_score IS 'Personality fit component score (0-100) for retention analysis';
COMMENT ON COLUMN resumes.retention_engagement_score IS 'Professional engagement component score (0-100) for retention analysis';
COMMENT ON COLUMN resumes.retention_fitment_factor IS 'Fitment factor component score (0-100) for retention analysis';
COMMENT ON COLUMN resumes.retention_institution_quality IS 'Institution quality component score (0-100) for retention analysis';

-- Create index for faster queries on component scores
CREATE INDEX IF NOT EXISTS idx_resumes_retention_stability ON resumes(retention_stability_score);
CREATE INDEX IF NOT EXISTS idx_resumes_retention_personality ON resumes(retention_personality_score);
CREATE INDEX IF NOT EXISTS idx_resumes_retention_engagement ON resumes(retention_engagement_score);


