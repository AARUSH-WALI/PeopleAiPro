# Job Selection Before Resume Upload - Feature Documentation

## Overview
This feature allows HR to select a job position before uploading a resume. The selected job is used to evaluate the candidate's fitment instead of the model guessing the job role.

## Changes Made

### Frontend Changes

1. **`client/src/components/resume/ResumeUpload.tsx`**
   - Added job selection dropdown that fetches active jobs on component mount
   - Requires job selection before allowing file upload/parsing
   - Modified Gemini prompt to use selected job details instead of guessing
   - Added `job_id` to the resume data mapping

2. **`client/src/types/resume.ts`**
   - Added `job_id?: string` to `ResumeData` interface

### Backend Changes

1. **`server/src/controllers/resumeController.js`**
   - Added `job_id` handling in the `createResume` function
   - Job ID is stored when provided in resume data

2. **`server/migrations/010_add_job_id_to_resumes.sql`**
   - Migration to add `job_id` column to `resumes` table
   - Creates foreign key relationship with `jobs` table
   - Adds index for better query performance

## How It Works

1. When the Resume Upload page loads, it fetches all active jobs from the API
2. HR must select a job from the dropdown before uploading a file
3. The selected job's title, description, and requirements are included in the Gemini prompt
4. The model evaluates the candidate specifically for the selected job position
5. The `job_id` is stored with the resume data in the database

## How to Undo This Feature

If you want to revert this feature, follow these steps:

### 1. Database Migration
```sql
-- Run this SQL to remove the job_id column
ALTER TABLE resumes DROP COLUMN IF EXISTS job_id;
DROP INDEX IF EXISTS idx_resumes_job_id;
```

### 2. Frontend Changes
In `client/src/components/resume/ResumeUpload.tsx`:
- Remove the job selection dropdown UI (lines with `Select` component)
- Remove `selectedJob` state and related logic
- Remove `jobs` state and `useEffect` for fetching jobs
- Remove job selection validation in `parseResume` function
- Revert the prompt to the original version (remove job-specific instructions)
- Remove `job_id` from resume data mapping

### 3. Backend Changes
In `server/src/controllers/resumeController.js`:
- Remove the line: `...(resumeData.job_id && { job_id: resumeData.job_id }),`

### 4. Type Definition
In `client/src/types/resume.ts`:
- Remove `job_id?: string;` from `ResumeData` interface

### 5. Delete Migration File
- Delete `server/migrations/010_add_job_id_to_resumes.sql` (optional, for clean history)

## Notes

- The feature is designed to be easily reversible
- All changes are documented with comments in the code
- The job selection is required - users cannot proceed without selecting a job
- If no active jobs exist, a helpful message is displayed


