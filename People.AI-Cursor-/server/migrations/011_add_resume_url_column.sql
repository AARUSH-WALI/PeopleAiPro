-- Add resume_url column to resumes table for Supabase Storage public URLs
-- This column will store the public URL of resumes stored in Supabase Storage

ALTER TABLE resumes
ADD COLUMN IF NOT EXISTS resume_url TEXT;

-- Add comment for documentation
COMMENT ON COLUMN resumes.resume_url IS 'Public URL of the resume file stored in Supabase Storage (resumes bucket)';

-- Create index for faster queries on resume_url
CREATE INDEX IF NOT EXISTS idx_resumes_resume_url ON resumes(resume_url);

