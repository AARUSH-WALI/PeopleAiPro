-- Add Big Five personality score columns to resumes table
-- Uses lowercase snake_case for consistency

ALTER TABLE resumes
ADD COLUMN IF NOT EXISTS extraversion INTEGER,
ADD COLUMN IF NOT EXISTS agreeableness INTEGER,
ADD COLUMN IF NOT EXISTS conscientiousness INTEGER,
ADD COLUMN IF NOT EXISTS neuroticism INTEGER,
ADD COLUMN IF NOT EXISTS openness INTEGER;

-- Optional: simple constraints (scores typically between -50 and 50 in your UI)
-- Adjust as needed or remove if not desired
ALTER TABLE resumes
  ADD CONSTRAINT IF NOT EXISTS chk_extraversion_range CHECK (extraversion IS NULL OR extraversion BETWEEN -100 AND 100),
  ADD CONSTRAINT IF NOT EXISTS chk_agreeableness_range CHECK (agreeableness IS NULL OR agreeableness BETWEEN -100 AND 100),
  ADD CONSTRAINT IF NOT EXISTS chk_conscientiousness_range CHECK (conscientiousness IS NULL OR conscientiousness BETWEEN -100 AND 100),
  ADD CONSTRAINT IF NOT EXISTS chk_neuroticism_range CHECK (neuroticism IS NULL OR neuroticism BETWEEN -100 AND 100),
  ADD CONSTRAINT IF NOT EXISTS chk_openness_range CHECK (openness IS NULL OR openness BETWEEN -100 AND 100);





