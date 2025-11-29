-- Create fitment_criteria table
CREATE TABLE IF NOT EXISTS fitment_criteria (
  -- Primary key
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  
  -- Fitment criteria thresholds
  best_fit INTEGER NOT NULL,
  average_fit INTEGER NOT NULL,
  not_fit INTEGER NOT NULL,
  
  -- Metadata
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  created_by UUID,
  
  -- Add constraints
  CONSTRAINT valid_fitment_thresholds CHECK (
    best_fit > average_fit AND 
    average_fit > not_fit AND
    not_fit >= 0
  )
);

-- Create a trigger to automatically update the updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_fitment_criteria_updated_at
BEFORE UPDATE ON fitment_criteria
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Insert default values if the table is empty
INSERT INTO fitment_criteria (best_fit, average_fit, not_fit, created_by)
SELECT 80, 50, 0, NULL
WHERE NOT EXISTS (SELECT 1 FROM fitment_criteria);

-- Create indexes for better query performance if needed
-- CREATE INDEX IF NOT EXISTS idx_fitment_criteria_created_at ON fitment_criteria(created_at);
