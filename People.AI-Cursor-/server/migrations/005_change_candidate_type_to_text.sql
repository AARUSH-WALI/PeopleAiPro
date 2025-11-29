-- Change candidate_type from boolean to text to store Experienced/Inexperienced/Fresher
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 
    FROM information_schema.columns 
    WHERE table_name='resumes' AND column_name='candidate_type' AND data_type='boolean'
  ) THEN
    ALTER TABLE resumes
    ALTER COLUMN candidate_type DROP DEFAULT;

    ALTER TABLE resumes
    ALTER COLUMN candidate_type TYPE text USING (
      CASE 
        WHEN candidate_type IS TRUE THEN 'Experienced'
        WHEN candidate_type IS FALSE THEN 'Fresher'
        ELSE NULL
      END
    );
  END IF;
END $$;






