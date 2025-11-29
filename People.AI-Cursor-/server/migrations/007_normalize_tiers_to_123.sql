-- Normalize existing tier data to only use 1, 2, 3, or NULL
-- This migration should be run after 006_add_retention_analysis.sql

-- Update any tiers that are 4 or 5 to 3
-- Update any NULL tiers: if institution exists, set to 3; if institution is empty, keep as NULL

UPDATE resumes
SET ug_tier = CASE
    WHEN ug_tier IS NULL AND (ug_institute IS NOT NULL AND ug_institute != '') THEN 3
    WHEN ug_tier IN (4, 5) THEN 3
    WHEN ug_tier NOT IN (1, 2, 3) THEN 3
    ELSE ug_tier
END
WHERE ug_tier IS NULL OR ug_tier NOT IN (1, 2, 3);

UPDATE resumes
SET pg_tier = CASE
    WHEN pg_tier IS NULL AND (pg_institute IS NOT NULL AND pg_institute != '') THEN 3
    WHEN pg_tier IN (4, 5) THEN 3
    WHEN pg_tier NOT IN (1, 2, 3) THEN 3
    ELSE pg_tier
END
WHERE pg_tier IS NULL OR pg_tier NOT IN (1, 2, 3);

-- Set to NULL if institution is empty
UPDATE resumes
SET ug_tier = NULL
WHERE (ug_institute IS NULL OR ug_institute = '') AND ug_tier IS NOT NULL;

UPDATE resumes
SET pg_tier = NULL
WHERE (pg_institute IS NULL OR pg_institute = '') AND pg_tier IS NOT NULL;



