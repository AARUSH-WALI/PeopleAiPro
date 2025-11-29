-- Enable UUID extension if not exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create interviews table
CREATE TABLE IF NOT EXISTS interviews (
  -- Primary key
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  
  -- Basic interview information
  title VARCHAR(255) NOT NULL,
  description TEXT,
  status VARCHAR(20) NOT NULL DEFAULT 'scheduled' CHECK (status IN ('scheduled', 'completed', 'cancelled')),
  
  -- Candidate information
  candidate_id UUID REFERENCES resumes(id) ON DELETE CASCADE,
  candidate_name VARCHAR(255) NOT NULL,
  candidate_email VARCHAR(255) NOT NULL,
  
  -- Interview timing
  date TIMESTAMP WITH TIME ZONE NOT NULL,
  scheduled_start_time TIMESTAMP WITH TIME ZONE,
  scheduled_end_time TIMESTAMP WITH TIME ZONE,
  actual_start_time TIMESTAMP WITH TIME ZONE,
  actual_end_time TIMESTAMP WITH TIME ZONE,
  
  -- Interviewer information
  interviewer_id UUID,
  interviewer_name VARCHAR(255),
  interviewer_email VARCHAR(255),
  
  -- Meeting details
  meeting_link TEXT,
  meeting_platform VARCHAR(100),
  
  -- Interview details
  interview_type VARCHAR(100),
  job_title VARCHAR(255),
  job_description TEXT,
  
  -- Feedback and notes
  interviewer_notes TEXT,
  candidate_feedback TEXT,
  technical_assessment TEXT,
  overall_rating INTEGER CHECK (overall_rating IS NULL OR (overall_rating >= 1 AND overall_rating <= 5)),
  
  -- Metadata
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  created_by UUID,
  
  -- Add constraints
  CONSTRAINT fk_candidate FOREIGN KEY (candidate_id) REFERENCES resumes(id) ON DELETE CASCADE
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_interviews_candidate_id ON interviews(candidate_id);
CREATE INDEX IF NOT EXISTS idx_interviews_interviewer_id ON interviews(interviewer_id);
CREATE INDEX IF NOT EXISTS idx_interviews_status ON interviews(status);
CREATE INDEX IF NOT EXISTS idx_interviews_scheduled_time ON interviews(scheduled_start_time, scheduled_end_time);

-- Create a trigger to automatically update the updated_at column
CREATE TRIGGER update_interviews_updated_at
BEFORE UPDATE ON interviews
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();
