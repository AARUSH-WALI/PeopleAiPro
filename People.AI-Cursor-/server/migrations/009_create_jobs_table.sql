-- Create jobs table for storing job listings
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    requirements TEXT[], -- Array of requirement strings
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for active jobs
CREATE INDEX IF NOT EXISTS idx_jobs_active ON jobs(active);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);

-- Add comment for documentation
COMMENT ON TABLE jobs IS 'Stores job listings with title, description, requirements, and active status';
COMMENT ON COLUMN jobs.active IS 'Whether the job listing is currently active/visible';
COMMENT ON COLUMN jobs.requirements IS 'Array of job requirement strings';


