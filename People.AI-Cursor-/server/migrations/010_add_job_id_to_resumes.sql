-- Migration: Add job_id column to resumes table
-- This migration adds support for associating resumes with specific job positions
-- 
-- FEATURE: Job Selection Before Resume Upload
-- This allows HR to select a job before uploading a resume, so the model
-- evaluates the candidate based on the selected job instead of guessing.
--
-- TO UNDO THIS FEATURE:
-- 1. Run: ALTER TABLE resumes DROP COLUMN IF EXISTS job_id;
-- 2. Remove job_id handling from resumeController.js
-- 3. Remove job selection UI from ResumeUpload.tsx
-- 4. Revert the prompt changes in ResumeUpload.tsx

-- Add job_id column as a foreign key to jobs table
ALTER TABLE resumes 
ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES jobs(id) ON DELETE SET NULL;

-- Create index for better query performance
CREATE INDEX IF NOT EXISTS idx_resumes_job_id ON resumes(job_id);

-- Add comment
COMMENT ON COLUMN resumes.job_id IS 'Foreign key to jobs table. Associates the resume with a specific job position for evaluation.';


