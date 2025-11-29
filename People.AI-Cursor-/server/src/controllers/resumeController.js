import { supabase } from '../config/supabase.js';
import { calculateOverallFitment, calculateDatasetScore } from '../services/fitmentService.js';
import { retentionScorer } from '../services/retentionService.js';
import { generateRetentionInsight } from '../services/retentionInsightsService.js';
import { v4 as uuidv4 } from 'uuid';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Upload file to Supabase Storage using candidate ID
// This function uploads files to the 'resumes' bucket with path: ${candidateId}/${file.name}
const uploadFileToStorage = async (file, candidateId) => {
  console.log('[uploadFileToStorage] Starting file upload:', {
    originalName: file.originalname,
    candidateId: candidateId,
    size: file.size,
    mimeType: file.mimetype
  });

  try {
    if (!file || !file.path) {
      throw new Error('No file or file path provided');
    }

    if (!candidateId) {
      throw new Error('Candidate ID is required for file upload');
    }

    // Sanitize filename to remove special characters
    const sanitizedFileName = file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_');
    const filePath = `${candidateId}/${sanitizedFileName}`;

    console.log('[uploadFileToStorage] Preparing to upload file to Supabase:', {
      originalName: file.originalname,
      sanitizedFileName: sanitizedFileName,
      tempPath: file.path,
      size: file.size,
      mimeType: file.mimetype,
      destinationPath: filePath,
      bucket: 'resumes'
    });

    // Note: Bucket 'resumes' should be created manually in Supabase dashboard
    console.log('[uploadFileToStorage] Assuming bucket "resumes" exists (create it manually in Supabase dashboard if not)');

    // Read the file as a buffer
    console.log('[uploadFileToStorage] Reading file from disk...');
    const fileBuffer = fs.readFileSync(file.path);

    if (!fileBuffer || fileBuffer.length === 0) {
      throw new Error('File is empty or could not be read');
    }

    console.log('[uploadFileToStorage] Uploading file to Supabase Storage...');
    // Upload with upsert: true to overwrite existing files
    const { data: uploadData, error: uploadError } = await supabase.storage
      .from('resumes')
      .upload(filePath, fileBuffer, {
        cacheControl: '3600',
        upsert: true, // Overwrite existing file if it exists
        contentType: file.mimetype,
        duplex: 'half'
      });

    if (uploadError) {
      console.error('[uploadFileToStorage] Error uploading file to Supabase:', {
        error: uploadError,
        message: uploadError.message,
        statusCode: uploadError.statusCode
      });

      // Provide specific error messages for common issues
      if (uploadError.message && uploadError.message.includes('Bucket not found')) {
        throw new Error('Storage bucket "resumes" not found. Please create it in your Supabase dashboard under Storage.');
      }
      if (uploadError.message && uploadError.message.includes('row-level security policy')) {
        throw new Error('Storage permissions error. Please check your Supabase RLS policies or use a service role key instead of anon key.');
      }
      if (uploadError.message && uploadError.message.includes('The resource already exists')) {
        // This shouldn't happen with upsert: true, but handle it just in case
        console.warn('[uploadFileToStorage] File already exists, but upsert should handle this');
      }

      throw new Error(`File upload failed: ${uploadError.message || 'Unknown error'}`);
    }

    console.log('[uploadFileToStorage] File uploaded successfully, getting public URL...');
    // Get public URL
    const { data: { publicUrl } } = supabase.storage
      .from('resumes')
      .getPublicUrl(filePath);

    if (!publicUrl) {
      throw new Error('Failed to get public URL for uploaded file');
    }

    console.log('[uploadFileToStorage] Public URL retrieved:', publicUrl);

    // Clean up the temporary file
    try {
      console.log('[uploadFileToStorage] Cleaning up temporary file...');
      fs.unlinkSync(file.path);
      console.log('[uploadFileToStorage] Temporary file removed successfully');
    } catch (cleanupError) {
      console.error('[uploadFileToStorage] Error cleaning up temporary file:', cleanupError);
      // Don't fail the request if cleanup fails
    }

    const result = {
      original_filename: file.originalname,
      file_path: filePath,
      resume_url: publicUrl, // This is the public URL we'll save to database
      file_size: file.size,
      mime_type: file.mimetype,
      uploaded_at: new Date().toISOString()
    };

    console.log('[uploadFileToStorage] File upload completed successfully:', {
      original_filename: result.original_filename,
      file_path: result.file_path,
      resume_url: result.resume_url,
      file_size: result.file_size,
      mime_type: result.mime_type
    });

    return result;
  } catch (error) {
    console.error('[uploadFileToStorage] Error in file upload process:', {
      error: error.message,
      stack: error.stack,
      candidateId: candidateId,
      file: file ? {
        originalname: file.originalname,
        path: file.path,
        size: file?.size,
        mimetype: file?.mimetype
      } : 'No file info'
    });
    throw error;
  }
};

// Get all resumes with pagination
export const getResumes = async (req, res) => {
  try {
    console.log('Fetching all resumes...');

    const { page = 1, limit = 10 } = req.query;
    const offset = (page - 1) * limit;

    const { data: resumes, error, count } = await supabase
      .from('resumes')
      .select('*', { count: 'exact' })
      .order('created_at', { ascending: false })
      .range(offset, offset + limit - 1);

    if (error) throw error;

    // Get API base URL from app.locals
    const apiBaseUrl = req.app.locals.API_BASE_URL || 'http://localhost:5000';

    // Transform the data to include full file URLs
    const resumesWithUrls = resumes.map(resume => ({
      ...resume,
      file_url: resume.file_path ? `${apiBaseUrl}/resumes/${resume.id}/file` : null
    }));

    res.json({
      success: true,
      data: resumesWithUrls,
      pagination: {
        total: count,
        page: parseInt(page),
        limit: parseInt(limit),
        totalPages: Math.ceil(count / limit)
      }
    });
  } catch (error) {
    console.error('Error getting resumes:', error);
    res.status(500).json({ success: false, error: error.message });
  }
};

// Search resumes
export const searchResumes = async (req, res) => {
  try {
    const { query } = req.query;

    if (!query) {
      return res.status(400).json({ error: 'Search query is required' });
    }

    const { data, error } = await supabase
      .from('resumes')
      .select('*')
      .or(`name.ilike.%${query}%,email.ilike.%${query}%,skills.cs.{"${query}"}`);

    if (error) throw error;

    res.json(data);
  } catch (error) {
    console.error('Error searching resumes:', error);
    res.status(500).json({ error: 'Failed to search resumes' });
  }
};

// Get a specific resume by ID
export const getResumeById = async (req, res) => {
  try {
    const { id } = req.params;
    console.log(`[getResumeById] Fetching resume with ID: ${id}`);

    const { data, error } = await supabase
      .from('resumes')
      .select('*')
      .eq('id', id)
      .single();

    if (error) {
      console.error(`[getResumeById] Error fetching resume:`, error);
      throw error;
    }

    if (!data) {
      console.warn(`[getResumeById] Resume not found for ID: ${id}`);
      return res.status(404).json({ success: false, error: 'Resume not found' });
    }

    console.log(`[getResumeById] Resume found:`, {
      id: data.id,
      name: data.name,
      email: data.email,
      resume_url: data.resume_url,
      file_url: data.file_url,
      file_path: data.file_path
    });

    // Get API base URL from app.locals
    const apiBaseUrl = req.app.locals.API_BASE_URL || 'http://localhost:5000';

    // Prefer resume_url (Supabase Storage public URL) over constructed file_url
    // Only construct file_url if resume_url is not available
    if (!data.resume_url && data.file_path) {
      data.file_url = `${apiBaseUrl}/resumes/${data.id}/file`;
    } else if (data.resume_url) {
      // If resume_url exists, also set it as file_url for backward compatibility
      data.file_url = data.resume_url;
    }

    res.json({ success: true, data });
  } catch (error) {
    console.error('Error getting resume:', error);
    res.status(500).json({ success: false, error: error.message });
  }
};

// Create a new resume
export const createResume = async (req, res) => {
  console.log('=== Starting createResume ===');
  console.log('Request body:', JSON.stringify(req.body, null, 2));
  console.log('Request file:', req.file ? {
    fieldname: req.file.fieldname,
    originalname: req.file.originalname,
    mimetype: req.file.mimetype,
    size: req.file.size,
    path: req.file.path
  } : 'No file uploaded');

  try {
    let resumeData = {};

    // Parse the JSON data from the form data
    if (req.body.data) {
      try {
        console.log('Parsing resume data from request body');
        resumeData = typeof req.body.data === 'string'
          ? JSON.parse(req.body.data)
          : req.body.data;

        // If personalInfo exists, extract its fields to the root level
        if (resumeData.personalInfo) {
          const { personalInfo, ...rest } = resumeData;
          resumeData = {
            ...personalInfo,
            ...rest
          };
        }

        console.log('Processed resume data:', JSON.stringify(resumeData, null, 2));
      } catch (e) {
        console.error('Error parsing resume data:', {
          error: e.message,
          stack: e.stack,
          data: req.body.data
        });
        return res.status(400).json({
          success: false,
          error: 'Invalid resume data format',
          message: e.message,
          details: e.message
        });
      }
    } else {
      console.log('No resume data found in request body');
      // Try to get data from request body directly if not in data field
      const { data: bodyData, file, ...rest } = req.body;
      if (Object.keys(rest).length > 0) {
        resumeData = rest;
        console.log('Using data from request body:', JSON.stringify(resumeData, null, 2));
      }
    }

    // Note: File upload will be handled AFTER inserting the resume to get the candidate ID
    // We'll store file metadata temporarily for later use
    const fileToUpload = req.file;
    if (fileToUpload) {
      console.log('[createResume] File will be uploaded after resume creation:', {
        originalname: fileToUpload.originalname,
        size: fileToUpload.size,
        mimetype: fileToUpload.mimetype
      });
      // Store file metadata but don't set file_path/file_url yet
      resumeData.original_filename = fileToUpload.originalname;
      resumeData.file_size = fileToUpload.size;
      resumeData.mime_type = fileToUpload.mimetype;
    } else {
      console.log('[createResume] No file uploaded with the request');
    }

    // Map and validate resume data to match database schema
    const mappedResumeData = {
      // Personal info (required fields)
      name: resumeData.name || resumeData.personalInfo?.name || '',
      email: resumeData.email || resumeData.personalInfo?.email || '',
      phone: resumeData.phone || resumeData.personalInfo?.phone || null,
      address: resumeData.address || resumeData.personalInfo?.address || null,
      summary: resumeData.summary || resumeData.personalInfo?.summary || null,

      // Arrays (JSONB fields)
      education: resumeData.education || [],
      experience: resumeData.experience || [],
      skills: resumeData.skills || [],
      achievements: resumeData.achievements || [],
      projects: resumeData.projects || [],
      research_papers: resumeData.research_papers || [],
      patents: resumeData.patents || [],
      books: resumeData.books || [],
      trainings: resumeData.trainings || [],
      workshops: resumeData.workshops || [],

      // Education institutes
      ug_institute: resumeData.ug_institute || resumeData.UG_InstituteName || null,
      pg_institute: resumeData.pg_institute || resumeData.PG_InstituteName || null,
      phd_institute: resumeData.phd_institute || (resumeData.PhD_Institute ? String(resumeData.PhD_Institute) : null),
      // Institution tiers (1, 2, or 3 only, or null if empty)
      // Normalize: only accept 1-3, treat anything else as null or 3
      ug_tier: (() => {
        const tier = resumeData.ug_tier ?? resumeData.UG_Tier;
        // Convert to integer and validate
        const tierNum = tier !== null && tier !== undefined ? Math.round(Number(tier)) : null;
        if (tierNum !== null && !isNaN(tierNum) && tierNum >= 1 && tierNum <= 3) {
          return tierNum;
        }
        // If institution exists but tier is invalid/empty, default to tier 3
        // If institution is empty, return null
        return (resumeData.ug_institute || resumeData.UG_InstituteName) ? 3 : null;
      })(),
      pg_tier: (() => {
        const tier = resumeData.pg_tier ?? resumeData.PG_Tier;
        // Convert to integer and validate
        const tierNum = tier !== null && tier !== undefined ? Math.round(Number(tier)) : null;
        if (tierNum !== null && !isNaN(tierNum) && tierNum >= 1 && tierNum <= 3) {
          return tierNum;
        }
        // If institution exists but tier is invalid/empty, default to tier 3
        // If institution is empty, return null
        return (resumeData.pg_institute || resumeData.PG_InstituteName) ? 3 : null;
      })(),

      // Metadata and analytics
      // Ensure all integer fields are properly converted (handle decimal strings from Gemini)
      longevity_years: Math.round(Number(resumeData.longevity_years || resumeData.Longevity_Years || 0)),
      number_of_jobs: Math.round(Number(resumeData.number_of_jobs || resumeData.No_of_Jobs || 0)),
      average_experience: Number(resumeData.average_experience || resumeData.Experience_Average || 0), // NUMERIC - can have decimals
      skills_count: Math.round(Number(resumeData.skills_count || resumeData.Skills_No || 0)),
      achievements_count: Math.round(Number(resumeData.achievements_count || resumeData.Achievements_No || 0)),
      trainings_count: Math.round(Number(resumeData.trainings_count || resumeData.Trainings_No || 0)),
      workshops_count: Math.round(Number(resumeData.workshops_count || resumeData.Workshops_No || 0)),
      projects_count: Math.round(Number(resumeData.projects_count || resumeData.Projects_No || 0)),
      research_papers_count: Math.round(Number(resumeData.research_papers_count || resumeData.Research_Papers_No || 0)),
      patents_count: Math.round(Number(resumeData.patents_count || resumeData.Patents_No || 0)),
      books_count: Math.round(Number(resumeData.books_count || resumeData.Books_No || 0)),
      is_jk: Boolean(resumeData.is_jk || resumeData.State_JK || false),
      best_fit_for: resumeData.best_fit_for || resumeData.Best_Fit_For || null,

      // Job association - links resume to specific job position for display/metadata
      ...(resumeData.job_id && { job_id: resumeData.job_id }),

      // File info (if uploaded) - file_path and resume_url will be set after upload
      ...(resumeData.original_filename && { original_filename: resumeData.original_filename }),
      ...(resumeData.file_size && { file_size: resumeData.file_size }),
      ...(resumeData.mime_type && { mime_type: resumeData.mime_type }),
      personality_analysis_completed: false
    };

    // Calculate dataset score (resume evaluation score before Big5)
    // This is the raw score out of 100 based on resume data only
    try {
      const candidateDataForDataset = {
        longevity_years: mappedResumeData.longevity_years || 0,
        average_experience: mappedResumeData.average_experience || 0,
        workshops_count: mappedResumeData.workshops_count || 0,
        trainings_count: mappedResumeData.trainings_count || 0,
        research_papers_count: mappedResumeData.research_papers_count || 0,
        patents_count: mappedResumeData.patents_count || 0,
        achievements_count: mappedResumeData.achievements_count || 0,
        books_count: mappedResumeData.books_count || 0,
        is_jk: mappedResumeData.is_jk || false,
        number_of_jobs: mappedResumeData.number_of_jobs || 0,
        ug_institute: mappedResumeData.ug_institute || null,
        pg_institute: mappedResumeData.pg_institute || null,
        phd_institute: mappedResumeData.phd_institute || null
      };

      const datasetScore = calculateDatasetScore(candidateDataForDataset);
      mappedResumeData.dataset_score = Math.round(datasetScore * 100) / 100;
      console.log('[createResume] Calculated dataset_score:', mappedResumeData.dataset_score);
    } catch (error) {
      console.warn('[createResume] Error calculating dataset_score:', error);
      // Continue without dataset_score if calculation fails
      mappedResumeData.dataset_score = null;
    }

    // Validate required fields
    if (!mappedResumeData.name || !mappedResumeData.email) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields',
        message: 'Name and email are required fields',
        details: 'Name and email are required fields'
      });
    }

    // Final validation: ensure all integer fields are actually integers (not decimals)
    // This is a safety check to catch any decimal values that might have slipped through
    const integerFields = [
      'longevity_years', 'number_of_jobs', 'skills_count', 'achievements_count',
      'trainings_count', 'workshops_count', 'projects_count', 'research_papers_count',
      'patents_count', 'books_count', 'ug_tier', 'pg_tier'
    ];

    // Also check for any other fields that might be integers (like personality scores)
    const allPossibleIntegerFields = [
      ...integerFields,
      'extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness'
    ];

    allPossibleIntegerFields.forEach(field => {
      if (mappedResumeData[field] !== null && mappedResumeData[field] !== undefined) {
        const value = mappedResumeData[field];
        // Check if it's a string with decimal or a number that's not an integer
        if (typeof value === 'string') {
          const numValue = Number(value);
          if (!isNaN(numValue) && !Number.isInteger(numValue)) {
            mappedResumeData[field] = Math.round(numValue);
            console.log(`[DataFix] Converted ${field} from string "${value}" to integer ${mappedResumeData[field]}`);
          } else if (!isNaN(numValue)) {
            mappedResumeData[field] = numValue; // Convert string number to actual number
          }
        } else if (typeof value === 'number' && !Number.isInteger(value)) {
          mappedResumeData[field] = Math.round(value);
          console.log(`[DataFix] Converted ${field} from decimal ${value} to integer ${mappedResumeData[field]}`);
        }
      }
    });

    // Additional safety: scan ALL fields for any decimal strings that might be integers
    Object.keys(mappedResumeData).forEach(key => {
      // Skip known numeric/decimal fields
      if (['average_experience', 'fitment_score', 'profile_score', 'retention_score',
        'retention_stability_score', 'retention_personality_score',
        'retention_engagement_score', 'retention_fitment_factor',
        'retention_institution_quality', 'dataset_score'].includes(key)) {
        return; // These can have decimals
      }

      const value = mappedResumeData[key];
      if (value !== null && value !== undefined && typeof value === 'string') {
        // Check if it looks like a decimal number
        if (/^-?\d+\.\d+$/.test(value.trim())) {
          // It's a decimal string - check if this field should be integer
          if (integerFields.includes(key) || key.includes('_count') || key.includes('_tier') ||
            key.includes('years') || key.includes('jobs') ||
            ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness'].includes(key)) {
            mappedResumeData[key] = Math.round(Number(value));
            console.log(`[DataFix] Auto-converted ${key} from "${value}" to integer ${mappedResumeData[key]}`);
          }
        }
      }
    });

    console.log('Inserting resume data into database...', JSON.stringify(mappedResumeData, null, 2));
    const { data: insertedData, error: dbError } = await supabase
      .from('resumes')
      .insert([mappedResumeData])
      .select();

    if (dbError) {
      console.error('Database insert error details:', {
        error: dbError,
        message: dbError.message,
        details: dbError.details,
        hint: dbError.hint,
        code: dbError.code
      });

      // Enhanced error logging: check which field has the issue
      if (dbError.message && dbError.message.includes('invalid input syntax for type integer')) {
        console.error('=== INTEGER TYPE ERROR DETECTED ===');
        console.error('Full mappedResumeData:', JSON.stringify(mappedResumeData, null, 2));

        // Check all fields for decimal values
        Object.keys(mappedResumeData).forEach(key => {
          const value = mappedResumeData[key];
          if (value !== null && value !== undefined) {
            if (typeof value === 'string' && value.includes('.')) {
              console.error(`⚠️ Field "${key}" has decimal string value: "${value}"`);
            } else if (typeof value === 'number' && !Number.isInteger(value)) {
              console.error(`⚠️ Field "${key}" has decimal number value: ${value}`);
            }
          }
        });
      }

      const errorMsg = `Database operation failed: ${dbError.message}${dbError.details ? ` - ${dbError.details}` : ''}${dbError.hint ? ` (Hint: ${dbError.hint})` : ''}`;
      throw new Error(errorMsg);
    }

    if (!insertedData || insertedData.length === 0) {
      throw new Error('No data returned from database after insert');
    }

    const candidateId = insertedData[0].id;
    console.log('[createResume] Resume created successfully with ID:', candidateId);

    // If file was uploaded, now upload it to Supabase Storage using the candidate ID
    let uploadResult = null;
    if (fileToUpload && candidateId) {
      try {
        console.log('[createResume] Uploading file to Supabase Storage...');
        uploadResult = await uploadFileToStorage(fileToUpload, candidateId);

        // Update the resume record with resume_url and file_path
        console.log('[createResume] Updating resume record with resume_url...');
        const { data: updatedData, error: updateError } = await supabase
          .from('resumes')
          .update({
            resume_url: uploadResult.resume_url,
            file_path: uploadResult.file_path,
            file_url: uploadResult.resume_url // Keep file_url for backward compatibility
          })
          .eq('id', candidateId)
          .select();

        if (updateError) {
          console.error('[createResume] Error updating resume with resume_url:', updateError);
          // Don't fail the request, but log the error
          // The resume was created successfully, just the file URL update failed
        } else {
          console.log('[createResume] Resume updated with resume_url:', uploadResult.resume_url);
          // Update insertedData with the new resume_url
          insertedData[0].resume_url = uploadResult.resume_url;
          insertedData[0].file_path = uploadResult.file_path;
          insertedData[0].file_url = uploadResult.resume_url;
        }
      } catch (uploadError) {
        console.error('[createResume] Error uploading file to Supabase Storage:', {
          error: uploadError.message,
          stack: uploadError.stack
        });
        // Don't fail the request if file upload fails
        // The resume was created successfully, just the file upload failed
        // Return error in response but with success status
      }
    }

    console.log('[createResume] Resume creation process completed:', JSON.stringify(insertedData[0], null, 2));

    // Prepare response data
    const responseData = {
      ...insertedData[0],
      resume_url: insertedData[0].resume_url || null,
      file_url: insertedData[0].resume_url || insertedData[0].file_url || null
    };

    // If file upload failed, include error in response
    const response = {
      success: true,
      message: 'Resume created successfully' + (uploadResult ? ' and file uploaded to Supabase Storage' : ''),
      data: responseData
    };

    if (fileToUpload && !uploadResult) {
      response.warning = 'Resume created but file upload failed. Please upload the file again.';
    }

    res.status(201).json(response);
  } catch (error) {
    console.error('Unexpected error in createResume:', {
      error: error.message,
      stack: error.stack,
      request: {
        method: req.method,
        url: req.originalUrl,
        headers: req.headers,
        body: req.body,
        file: req.file ? {
          originalname: req.file.originalname,
          mimetype: req.file.mimetype,
          size: req.file.size
        } : null
      }
    });

    const errorResponse = {
      success: false,
      error: 'Failed to create resume',
      message: error.message || 'An unknown error occurred',
      ...(process.env.NODE_ENV === 'development' && {
        stack: error.stack,
        details: error.details
      })
    };

    console.error('Sending error response:', errorResponse);
    res.status(500).json(errorResponse);
  } finally {
    console.log('=== End of createResume ===');
  }
};

// Update a resume
export const updateResume = async (req, res) => {
  try {
    const { id } = req.params;
    const updateData = req.body;

    // Handle file upload if present
    if (req.file) {
      const fileData = await uploadFileToStorage(req.file);
      updateData.file_data = fileData;
    }

    const { data, error } = await supabase
      .from('resumes')
      .update(updateData)
      .eq('id', id)
      .select();

    if (error) throw error;
    if (!data || data.length === 0) {
      return res.status(404).json({ error: 'Resume not found' });
    }

    res.json(data[0]);
  } catch (error) {
    console.error('Error updating resume:', error);
    res.status(500).json({ error: 'Failed to update resume' });
  }
};

// Delete a resume
export const deleteResume = async (req, res) => {
  try {
    const { id } = req.params;
    const { error } = await supabase
      .from('resumes')
      .delete()
      .eq('id', id);

    if (error) throw error;

    res.status(204).send();
  } catch (error) {
    console.error('Error deleting resume:', error);
    res.status(500).json({ error: 'Failed to delete resume' });
  }
};

// Update personality test results by email
export const updatePersonalityTestResults = async (req, res) => {
  try {
    const { email, extraversion, agreeableness, conscientiousness, neuroticism, openness } = req.body;
    console.log('[PersonalityTest] Incoming update for:', email);

    // Validate required fields
    if (!email || !email.trim()) {
      return res.status(400).json({
        success: false,
        error: 'Email is required',
        message: 'Email parameter is missing or empty'
      });
    }

    // Validate scores are numbers
    const scores = {
      extraversion: parseInt(extraversion),
      agreeableness: parseInt(agreeableness),
      conscientiousness: parseInt(conscientiousness),
      neuroticism: parseInt(neuroticism),
      openness: parseInt(openness)
    };

    if (
      isNaN(scores.extraversion) ||
      isNaN(scores.agreeableness) ||
      isNaN(scores.conscientiousness) ||
      isNaN(scores.neuroticism) ||
      isNaN(scores.openness)
    ) {
      return res.status(400).json({
        success: false,
        error: 'Invalid score format',
        message: 'All scores must be valid numbers'
      });
    }

    // Update data matching database column names
    // Use lowercase snake_case columns
    const updateData = {
      extraversion: scores.extraversion,
      agreeableness: scores.agreeableness,
      conscientiousness: scores.conscientiousness,
      neuroticism: scores.neuroticism,
      openness: scores.openness,
      personality_analysis_completed: true
    };

    console.log('Updating personality test results:', {
      email: email.toLowerCase().trim(),
      scores: updateData
    });

    // Update the resume record matching the email
    const { data, error } = await supabase
      .from('resumes')
      .update(updateData)
      .eq('email', email.toLowerCase().trim())
      .select();

    if (error) {
      console.error('Database update error:', error);
      return res.status(500).json({
        success: false,
        error: 'Database operation failed',
        message: error.message
      });
    }

    if (!data || data.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Resume not found',
        message: `No resume found with email: ${email}`
      });
    }

    // Recompute fitment using current record and new Big5 scores
    const record = data[0];
    console.log('[PersonalityTest] Recomputing fitment for:', record.email);

    // If job_id exists, preserve the job-specific fitment_score from Gemini
    // Only adjust it slightly based on personality, but keep it job-specific
    const hasJobSpecificScore = record.job_id && record.fitment_score;

    // Calculate fitment for personality score and category (always needed)
    const fitment = calculateOverallFitment(
      {
        longevity_years: record.longevity_years,
        average_experience: record.average_experience,
        workshops_count: record.workshops_count,
        trainings_count: record.trainings_count,
        research_papers_count: record.research_papers_count,
        patents_count: record.patents_count,
        achievements_count: record.achievements_count,
        books_count: record.books_count,
        is_jk: record.is_jk,
        number_of_jobs: record.number_of_jobs,
        ug_institute: record.ug_institute,
        pg_institute: record.pg_institute,
        phd_institute: record.phd_institute
      },
      {
        extraversion: scores.extraversion,
        agreeableness: scores.agreeableness,
        conscientiousness: scores.conscientiousness,
        neuroticism: scores.neuroticism,
        openness: scores.openness
      }
    );

    const personalityScore = Math.round(fitment.big5_score);
    const candidateCategory = fitment.category;

    // Always use the ML-based overall fitment score (dataset + Big5) for this candidate
    const overall = fitment.overall_fitment_score;

    // Normalize candidate_type to Title Case among allowed values
    const allowedTypes = new Set(['Experienced', 'Intermediate', 'Fresher']);
    const normalizedType = typeof candidateCategory === 'string'
      ? (() => {
        const t = candidateCategory.trim().toLowerCase();
        if (t === 'experienced') return 'Experienced';
        if (t === 'intermediate' || t === 'inexperienced') return 'Intermediate';
        if (t === 'fresher') return 'Fresher';
        return 'Fresher';
      })()
      : 'Fresher';
    const candidateType = allowedTypes.has(normalizedType) ? normalizedType : 'Fresher';

    // Calculate retention risk analysis
    console.log('[PersonalityTest] Calculating retention risk for:', record.email);

    // Prepare candidate data for retention calculation
    const candidateDataForRetention = {
      longevity_years: record.longevity_years || 0,
      number_of_unique_designations: record.number_of_unique_designations || record.number_of_jobs || 1,
      average_experience: record.average_experience || record.longevity_years || 1,
      workshops: record.workshops_count || record.workshops || 0,
      trainings: record.trainings_count || record.trainings || 0,
      total_papers: record.research_papers_count || record.total_papers || 0,
      total_patents: record.patents_count || record.total_patents || 0,
      achievements: record.achievements_count || record.achievements || 0,
      fitment_score: overall,
      UG_Tier: record.ug_tier || 3,
      PG_Tier: record.pg_tier || 3,
      category: candidateType,
      candidate_type: candidateType
    };

    // Prepare Big5 scores for retention calculation
    const big5ScoresForRetention = {
      conscientiousness: scores.conscientiousness,
      agreeableness: scores.agreeableness,
      neuroticism: scores.neuroticism,
      extroversion: scores.extraversion,
      openness: scores.openness
    };

    // Calculate retention risk
    const retentionAnalysis = retentionScorer.calculateRetentionRisk(
      candidateDataForRetention,
      overall,
      big5ScoresForRetention,
      candidateType,
      null, // peerGroup - will be auto-determined
      record.id // candidateId
    );

    console.log('[PersonalityTest] Retention analysis:', {
      retention_score: retentionAnalysis.retention_score,
      retention_risk: retentionAnalysis.retention_risk,
      peer_group: retentionAnalysis.peer_group
    });

    // Update database with fitment and retention data
    // Note: dataset_score is NOT updated here - it's only set when resume is created
    // fitment_score stores the overall score (dataset + Big5), profile_score is rounded for UI
    const updatePayload = {
      fitment_score: overall,
      profile_score: Math.round(overall),
      personality_score: personalityScore,
      candidate_type: candidateType,
      // Retention scores
      retention_score: retentionAnalysis.retention_score,
      retention_risk: retentionAnalysis.retention_risk,
      retention_analysis: retentionAnalysis,
      // Component scores (individual columns)
      retention_stability_score: retentionAnalysis.component_scores.stability,
      retention_personality_score: retentionAnalysis.component_scores.personality,
      retention_engagement_score: retentionAnalysis.component_scores.engagement,
      retention_fitment_factor: retentionAnalysis.component_scores.fitment_factor,
      retention_academic_tier_score: retentionAnalysis.component_scores.tier_score,
      retention_peer_comp_score: retentionAnalysis.component_scores.peer_comparison
    };

    const { data: updated, error: updError } = await supabase
      .from('resumes')
      .update(updatePayload)
      .eq('email', email.toLowerCase().trim())
      .select();

    if (updError) {
      console.error('Error updating fitment and retention fields:', updError);
      return res.status(500).json({
        success: false,
        error: 'Database operation failed',
        message: updError.message || 'Failed to update fitment and retention fields'
      });
    }

    console.log('Successfully updated personality test results, fitment score, and retention analysis:', updated?.[0] || record);
    res.status(200).json({
      success: true,
      message: 'Personality test results updated, fitment score computed, and retention analysis completed successfully',
      data: updated?.[0] || record
    });
  } catch (error) {
    console.error('Unexpected error updating personality test results:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      message: error.message || 'An unexpected error occurred'
    });
  }
};

// Get retention insights for a candidate
export const getRetentionInsights = async (req, res) => {
  try {
    const { id } = req.params;
    console.log(`[getRetentionInsights] ===== NEW REQUEST =====`);
    console.log(`[getRetentionInsights] Fetching retention insights for resume ID: ${id}`);
    console.log(`[getRetentionInsights] Request URL: ${req.originalUrl}`);

    // Get resume data including retention analysis
    const { data, error } = await supabase
      .from('resumes')
      .select('retention_score, retention_risk, retention_analysis')
      .eq('id', id)
      .single();

    if (error) {
      console.error(`[getRetentionInsights] Error fetching resume:`, error);
      throw error;
    }

    if (!data) {
      return res.status(404).json({ success: false, error: 'Resume not found' });
    }

    // Check if retention data exists
    if (!data.retention_analysis && !data.retention_score) {
      return res.status(404).json({
        success: false,
        error: 'Retention analysis not available for this candidate'
      });
    }

    // Parse retention_analysis if it's a string
    let retentionAnalysis = data.retention_analysis;
    if (typeof retentionAnalysis === 'string') {
      try {
        retentionAnalysis = JSON.parse(retentionAnalysis);
      } catch (e) {
        console.error('[getRetentionInsights] Error parsing retention_analysis:', e);
        retentionAnalysis = null;
      }
    }

    // Prepare retention data for Gemini
    const retentionData = {
      retention_score: data.retention_score || 0,
      retention_risk: data.retention_risk || 'Medium',
      component_scores: retentionAnalysis?.component_scores || {},
      risk_flags: retentionAnalysis?.risk_flags || []
    };

    // Get Gemini API key from environment
    const geminiApiKey = process.env.GEMINI_API_KEY;

    // Generate insight using Gemini (always fresh, no caching)
    console.log(`[getRetentionInsights] Calling Gemini API with retention data:`, {
      retention_score: retentionData.retention_score,
      retention_risk: retentionData.retention_risk,
      has_api_key: !!geminiApiKey
    });

    const insight = await generateRetentionInsight(retentionData, geminiApiKey);

    console.log(`[getRetentionInsights] Generated insight: "${insight}"`);

    res.json({
      success: true,
      insight: insight,
      timestamp: new Date().toISOString() // Add timestamp to ensure freshness
    });
  } catch (error) {
    console.error('Error getting retention insights:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to generate retention insights'
    });
  }
};

// Get resume file by ID
export const getResumeFile = async (req, res) => {
  try {
    const { id } = req.params;
    console.log(`Getting file for resume ID: ${id}`);

    // Get resume data from database
    const { data, error } = await supabase
      .from('resumes')
      .select('*')
      .eq('id', id)
      .single();

    if (error) throw error;
    if (!data) {
      return res.status(404).json({ success: false, error: 'Resume not found' });
    }

    // Check if file path exists
    if (!data.file_path) {
      return res.status(404).json({ success: false, error: 'No file associated with this resume' });
    }

    console.log(`File path in database: ${data.file_path}`);

    // 1) Try to serve from local uploads if path is local
    const maybeLocal = data.file_path.includes('uploads/') || data.file_path.includes('uploads\\');
    if (maybeLocal) {
      const localPath = path.join(path.dirname(fileURLToPath(import.meta.url)), '../../', data.file_path);
      try {
        if (fs.existsSync(localPath)) {
          const ext = path.extname(localPath).toLowerCase();
          let contentType = 'application/octet-stream';
          if (ext === '.pdf') contentType = 'application/pdf';
          else if (ext === '.doc') contentType = 'application/msword';
          else if (ext === '.docx') contentType = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
          else if (ext === '.txt') contentType = 'text/plain';

          res.setHeader('Content-Type', contentType);
          res.setHeader('Content-Disposition', `inline; filename="${path.basename(localPath)}"`);
          const stream = fs.createReadStream(localPath);
          stream.pipe(res);
          return;
        }
      } catch (e) {
        console.warn('Local file read failed, falling back to storage:', e.message);
      }
    }

    // 2) Fallback: Download the file from Supabase Storage
    try {
      const { data: fileData, error: downloadError } = await supabase.storage
        .from('resume')
        .download(data.file_path);

      if (downloadError) {
        console.error('Error downloading file from Supabase:', downloadError);
        return res.status(404).json({ success: false, error: 'File not found in storage' });
      }

      if (!fileData) {
        return res.status(404).json({ success: false, error: 'File data is empty' });
      }

      const ext = path.extname(data.file_path).toLowerCase();
      let contentType = 'application/octet-stream';
      if (ext === '.pdf') contentType = 'application/pdf';
      else if (ext === '.doc') contentType = 'application/msword';
      else if (ext === '.docx') contentType = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
      else if (ext === '.txt') contentType = 'text/plain';
      res.setHeader('Content-Type', contentType);
      res.setHeader('Content-Disposition', `inline; filename="${path.basename(data.file_path)}"`);
      const buffer = await fileData.arrayBuffer();
      res.send(Buffer.from(buffer));
    } catch (e) {
      console.error('Error serving file:', e);
      res.status(500).json({ success: false, error: e.message });
    }
  } catch (error) {
    console.error('Error getting resume file:', error);
    res.status(500).json({ success: false, error: error.message });
  }
};
