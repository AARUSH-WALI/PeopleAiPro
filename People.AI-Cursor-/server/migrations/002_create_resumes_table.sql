-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create a single table for all resume information
CREATE TABLE IF NOT EXISTS resumes (
  -- Primary key
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  
  -- File storage information
  original_filename VARCHAR(255),
  file_path TEXT,
  file_url TEXT,
  file_size BIGINT,
  mime_type VARCHAR(100),
  
  -- Personal Information
  name VARCHAR(255) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  phone VARCHAR(50),
  address TEXT,
  summary TEXT,
  
  -- Arrays stored as JSONB
  education JSONB DEFAULT '[]',
  experience JSONB DEFAULT '[]',
  skills JSONB DEFAULT '[]',
  achievements JSONB DEFAULT '[]',
  projects JSONB DEFAULT '[]',
  research_papers JSONB DEFAULT '[]',
  patents JSONB DEFAULT '[]',
  books JSONB DEFAULT '[]',
  trainings JSONB DEFAULT '[]',
  workshops JSONB DEFAULT '[]',
  
  -- Education Institutes
  ug_institute VARCHAR(255),
  pg_institute VARCHAR(255),
  phd_institute VARCHAR(100),
  
  -- Metadata and Analytics
  longevity_years INTEGER,
  number_of_jobs INTEGER,
  average_experience NUMERIC(3,1),
  skills_count INTEGER,
  achievements_count INTEGER,
  trainings_count INTEGER,
  workshops_count INTEGER,
  projects_count INTEGER,
  research_papers_count INTEGER,
  patents_count INTEGER,
  books_count INTEGER,
  is_jk BOOLEAN DEFAULT FALSE,
  best_fit_for VARCHAR(255),
  
  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_resumes_email ON resumes(email);
CREATE INDEX IF NOT EXISTS idx_resumes_best_fit ON resumes(best_fit_for);
CREATE INDEX IF NOT EXISTS idx_resumes_file_path ON resumes(file_path);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create a trigger to automatically update the updated_at column
CREATE TRIGGER update_resumes_updated_at
BEFORE UPDATE ON resumes
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();
