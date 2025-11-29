-- Add peer comparison and academic tier score columns
-- These columns match the new retention scorer implementation

ALTER TABLE resumes
ADD COLUMN IF NOT EXISTS retention_academic_tier_score NUMERIC(5,2),
ADD COLUMN IF NOT EXISTS retention_peer_comp_score NUMERIC(5,2);

-- Rename retention_institution_quality to retention_academic_tier_score if it exists
-- (for consistency with new naming)
DO $$
BEGIN
    -- Check if old column exists and new column doesn't
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'resumes' 
        AND column_name = 'retention_institution_quality'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'resumes' 
        AND column_name = 'retention_academic_tier_score'
    ) THEN
        -- Copy data from old column to new column
        UPDATE resumes 
        SET retention_academic_tier_score = retention_institution_quality;
        
        -- Drop old column (optional - comment out if you want to keep both)
        -- ALTER TABLE resumes DROP COLUMN retention_institution_quality;
    END IF;
END $$;

-- Add comments for documentation
COMMENT ON COLUMN resumes.retention_academic_tier_score IS 'Academic tier component score (0-100) for retention analysis';
COMMENT ON COLUMN resumes.retention_peer_comp_score IS 'Peer comparison component score (0-100) for retention analysis - calculated from historical data';

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_resumes_retention_academic_tier ON resumes(retention_academic_tier_score);
CREATE INDEX IF NOT EXISTS idx_resumes_retention_peer_comp ON resumes(retention_peer_comp_score);


