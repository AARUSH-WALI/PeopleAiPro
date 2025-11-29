import { useState, useEffect } from "react";
import { Upload, FileText, File as FileIcon, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ResumeData } from "@/types/resume";
import axios from "axios";
import { toast } from "sonner";

const API_BASE_URL = import.meta.env.PROD
  ? import.meta.env.VITE_PROD_API_BASE_URL
  : import.meta.env.VITE_DEV_API_BASE_URL || 'http://localhost:5000/api';

interface Job {
  id: string;
  title: string;
  active: boolean;
  description?: string;
  requirements?: string[];
}

interface ResumeUploadProps {
  onResumeUploaded: (data: ResumeData, file: File) => void;
  onParsingStateChange: (loading: boolean) => void;
}

export default function ResumeUpload({ onResumeUploaded, onParsingStateChange }: ResumeUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<string>("");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loadingJobs, setLoadingJobs] = useState(true);

  // Fetch active jobs on component mount
  useEffect(() => {
    const fetchActiveJobs = async () => {
      try {
        setLoadingJobs(true);
        const response = await axios.get(`${API_BASE_URL}/jobs`, {
          params: { active: 'true' }
        });
        const jobsData = response.data?.data || [];
        const activeJobs = jobsData
          .filter((job: any) => job.active !== false)
          .map((job: any) => ({
            id: job.id,
            title: job.title,
            active: job.active ?? true,
            description: job.description || '',
            requirements: Array.isArray(job.requirements) ? job.requirements : []
          }));
        setJobs(activeJobs);
      } catch (error: any) {
        console.error('Error fetching jobs:', error);
        toast.error('Failed to load jobs: ' + (error.response?.data?.error || error.message));
      } finally {
        setLoadingJobs(false);
      }
    };

    fetchActiveJobs();
  }, []);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      validateAndSetFile(droppedFile);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (uploadedFile: File) => {
    setError(null);

    // Check file type
    const validTypes = ["application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"];
    if (!validTypes.includes(uploadedFile.type)) {
      setError("Please upload a valid PDF or Word document");
      return;
    }

    // Check file size (10MB max)
    if (uploadedFile.size > 10 * 1024 * 1024) {
      setError("File size exceeds 10MB limit");
      return;
    }

    setFile(uploadedFile);
  };

  const getFileIcon = () => {
    if (!file) return <Upload size={40} />;

    switch (file.type) {
      case "application/pdf":
        return <FileText size={40} className="text-red-500" />;
      case "application/msword":
      case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return <FileIcon size={40} className="text-blue-500" />;
      default:
        return <FileIcon size={40} />;
    }
  };

  // Function to extract email from text or hyperlinks
  const extractEmailFromText = (text: string): string | null => {
    // Check for mailto: links first
    const mailtoRegex = /mailto:([^"'?]+)/i;
    const mailtoMatch = text.match(mailtoRegex);

    if (mailtoMatch && mailtoMatch[1]) {
      return mailtoMatch[1];
    }

    // Regular expression for matching email addresses
    const emailRegex = /([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)/gi;
    const matches = text.match(emailRegex);

    if (matches && matches.length > 0) {
      return matches[0];
    }

    return null;
  };

  const parseResume = async () => {
    if (!file) return;

    // Require job selection before parsing
    if (!selectedJob) {
      setError("Please select a job before parsing the resume");
      toast.error("Please select a job before parsing the resume");
      return;
    }

    try {
      setUploading(true);
      onParsingStateChange(true);
      setError(null);

      // Start progress indication
      const interval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(interval);
            return 90;
          }
          return prev + 10;
        });
      }, 300);

      // Convert file to base64 for sending to Gemini API
      const fileBase64 = await readFileAsBase64(file);

      const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
      if (!apiKey) {
        throw new Error('Gemini API key is not configured. Please set VITE_GEMINI_API_KEY in your environment.');
      }

      // Get selected job details for job-specific evaluation
      const selectedJobData = jobs.find(j => j.id === selectedJob);
      const jobTitle = selectedJobData?.title || '';
      const jobDescription = selectedJobData?.description || '';
      const jobRequirements = selectedJobData?.requirements || [];

      const geminiEndpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;

      // Enhanced prompt for job-specific candidate evaluation
      const prompt = `
        You are an expert HR evaluator. Your task is to extract structured information from a resume and evaluate the candidate specifically for a target job position.
        
        ============================================
        TARGET JOB POSITION FOR EVALUATION:
        ============================================
        Job Title: ${jobTitle}
        ${jobDescription ? `Job Description: ${jobDescription}` : ''}
        ${jobRequirements.length > 0 ? `Required Skills/Qualifications: ${jobRequirements.join(', ')}` : ''}
        
        ============================================
        EVALUATION INSTRUCTIONS:
        ============================================
        1. Extract all resume information accurately
        2. Compare the candidate's profile against the target job requirements above
        3. Assess skill alignment, experience relevance, and educational fit for THIS SPECIFIC JOB
        4. Calculate Fitment_Score based on how well the candidate matches THIS job (not a generic role)
        5. Use the exact job title "${jobTitle}" for Best_Fit_For field
        
        ============================================
        EXTRACT THE FOLLOWING FIELDS:
        ============================================
        
        - personalInfo: object with name, email, phone, address, summary
        - education: array of objects with institution, degree, field, startDate, endDate, gpa
        - experience: array of objects with company, position, startDate, endDate, description, location
        - skills: array of skills as strings
        - UG_InstituteName: string (undergraduate institution name)
        - PG_InstituteName: string (postgraduate institution name)
        - UG_Tier: number (institution tier: 1 for top-tier, 2 for good institutions, 3 for others. Return null/empty if institution name is empty. Only use 1, 2, or 3 - nothing else)
        - PG_Tier: number (institution tier: 1 for top-tier, 2 for good institutions, 3 for others. Return null/empty if institution name is empty. Only use 1, 2, or 3 - nothing else)
        - PhD_Institute: number (0 for no, 1 for yes)
        
        For UG_Tier and PG_Tier classification:
        - Tier 1: Top-tier institutions (IITs, IIMs, AIIMS, IISc, top central/state universities, premier engineering/medical colleges)
        - Tier 2: Good institutions (NITs, good state universities, reputable private universities, established engineering/medical colleges)
        - Tier 3: Other institutions (local colleges, lesser-known institutions, or if uncertain about the institution's reputation)
        - If institution name is empty or not found, do not include the tier field (or set to null)
        
        - Longevity_Years: number (working years count)
        - No_of_Jobs: number
        - Experience_Average: number (Longevity_Years / No_of_Jobs)
        - Skills_No: number
        - Achievements_No: number
        - Achievements: array of strings
        - Trainings_No: number
        - Trainings: array of strings
        - Workshops_No: number
        - Workshops: array of strings
        - Research_Papers_No: number
        - Research_Papers: array of strings
        - Patents_No: number
        - Patents: array of strings
        - Books_No: number
        - Books: array of strings
        - State_JK: number (0 for no, 1 for yes - for J&K resident)
        - Projects_No: number
        - Projects: array of strings
        
        - Best_Fit_For: string (MUST be exactly: "${jobTitle}" - this is the job being evaluated for)
        
        - Profile_Score: number (calculate based on the following criteria: 
          - pg_institute: 0.1 (1 if true, else 0)
          - phd_institute: 0.2 (1 if true, else 0)
          - longevity_years: 0.1 (max 5 years → normalized as min(value / 5, 1))
          - achievements_count: 0.05 (max 15 → normalized as min(value / 15, 1))
          - trainings_count: 0.05 (max 10 → normalized as min(value / 10, 1))
          - workshops_count: 0.05 (max 10 → normalized as min(value / 10, 1))
          - research_papers_count: 0.2 (max 10 → normalized as min(value / 10, 1))
          - patents_count: 0.1 (max 5 → normalized as min(value / 5, 1))
          - books_count: 0.05 (max 5 → normalized as min(value / 5, 1))
          - is_jk: 0.1 (1 if true, else 0)
          Multiply the sum by 100 to get Profile_Score
        ) 
        
        - Fitment_Score: number (CRITICAL: This MUST be job-specific and DIFFERENT for different jobs. Calculate using this EXACT formula:
          
          STEP 1: Calculate Base Score from Profile_Score (already calculated above)
          baseScore = Profile_Score
          
          STEP 2: Job-Specific Skill Match Analysis
          Required skills for "${jobTitle}": ${jobRequirements.length > 0 ? jobRequirements.join(', ') : 'See job description'}
          Candidate skills: [extract from resume]
          
          skillMatchScore = 0
          For each required skill:
            - If candidate has exact match: +3 points
            - If candidate has similar/related skill: +1.5 points
            - If candidate lacks skill: 0 points
          skillMatchScore = (matched_skills / total_required_skills) * 25
          (Maximum 25 points, minimum 0)
          
          STEP 3: Experience Relevance for "${jobTitle}"
          Analyze candidate's work experience:
          - If experience directly matches "${jobTitle}" role: +20 points
          - If experience is in related/similar field: +10 points
          - If experience is in different field but transferable: +5 points
          - If no relevant experience: 0 points
          experienceScore = calculated based on above (Maximum 20 points)
          
          STEP 4: Educational Background Suitability
          Check if candidate's education aligns with "${jobTitle}" requirements:
          - If degree/field directly matches job requirements: +10 points
          - If degree/field is related: +5 points
          - If degree/field is different: 0 points
          educationScore = calculated based on above (Maximum 10 points)
          
          STEP 5: Additional Qualifications Match
          Check certifications, trainings, projects against job description:
          - Highly relevant certifications/trainings: +5 points
          - Somewhat relevant: +2 points
          - Not relevant: 0 points
          additionalScore = calculated based on above (Maximum 5 points)
          
          STEP 6: Calculate Penalties for Major Gaps
          penalty = 0
          - If missing critical required skill: -10 points per critical skill
          - If experience level is too low for role: -15 points
          - If education doesn't meet minimum requirements: -10 points
          penalty = sum of all penalties (Maximum -30 points)
          
          STEP 7: Final Fitment_Score Calculation
          Fitment_Score = baseScore + skillMatchScore + experienceScore + educationScore + additionalScore + penalty
          
          IMPORTANT RULES:
          - Fitment_Score MUST be between 0 and 100
          - Fitment_Score MUST be different for different job roles
          - If candidate is evaluated for "Cybersecurity Analyst", the score should reflect cybersecurity fit
          - If same candidate is evaluated for "Machine Learning Engineer", the score should reflect ML fit
          - These two scores MUST be different based on skill/experience match
          
          Example: If candidate has ML skills but not cybersecurity skills:
            - For ML Engineer role: Fitment_Score should be HIGHER (e.g., 75-85)
            - For Cybersecurity Analyst role: Fitment_Score should be LOWER (e.g., 45-60)
        )
        
        For locations, check if it mentions Jammu, Kashmir, or J&K and set State_JK to 1 if it does.
        
        Return only valid JSON with no additional explanation or markdown formatting. Use double quotes for all keys and string values.
      `;

      try {
        // Make the actual API call to Gemini
        const response = await fetch(geminiEndpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            contents: [
              {
                parts: [
                  { text: prompt },
                  { inlineData: { mimeType: file.type, data: fileBase64 } }
                ]
              }
            ],
            generationConfig: {
              temperature: 0.1,
              maxOutputTokens: 4096,
              topP: 0.95,
              topK: 40
            }
          })
        });

        if (!response.ok) {
          let errorDetail = '';
          try {
            const errJson = await response.json();
            errorDetail = errJson.error?.message || JSON.stringify(errJson);
          } catch {
            errorDetail = await response.text();
          }
          throw new Error(`Gemini API request failed (${response.status}): ${errorDetail}`);
        }

        const data = await response.json();

        if (data.error) {
          throw new Error(data.error.message || 'Gemini API returned an error while parsing the resume.');
        }

        if (!data.candidates || !Array.isArray(data.candidates) || !data.candidates[0]) {
          throw new Error('Gemini API did not return any candidates. Please try again with a different resume.');
        }

        const firstCandidate = data.candidates[0];
        const firstPart = firstCandidate.content?.parts?.[0];
        if (!firstPart || typeof firstPart.text !== 'string') {
          throw new Error('Gemini response did not contain the expected text output.');
        }

        // Extract JSON from the response
        const parsedContent = firstPart.text;

        // Try multiple regex patterns to extract the JSON
        const jsonMatches = [
          // Match JSON enclosed in code blocks
          parsedContent.match(/```(?:json)?\s*([\s\S]*?)\s*```/),
          // Match JSON enclosed in brackets
          parsedContent.match(/(\{[\s\S]*\})/),
          // Fallback to the whole response
          { 1: parsedContent }
        ];

        // Find the first successful match
        let jsonText = null;
        for (const match of jsonMatches) {
          if (match && match[1]) {
            jsonText = match[1].trim();
            break;
          }
        }

        if (!jsonText) {
          throw new Error("Could not extract JSON from the response");
        }

        // Parse the JSON
        let parsedResumeData;
        try {
          parsedResumeData = JSON.parse(jsonText);
          console.log("Parsed resume data:", parsedResumeData);

          // Map received data to match ResumeData interface
          const mappedData: ResumeData = {
            personalInfo: parsedResumeData.personalInfo || {
              name: "",
              email: "",
              phone: "",
              address: "",
              summary: ""
            },
            education: parsedResumeData.education || [],
            experience: parsedResumeData.experience || [],
            skills: parsedResumeData.skills || [],
            ug_institute: parsedResumeData.UG_InstituteName || "",
            pg_institute: parsedResumeData.PG_InstituteName || "",
            ug_tier: (parsedResumeData.UG_Tier !== undefined && parsedResumeData.UG_Tier !== null && parsedResumeData.UG_Tier >= 1 && parsedResumeData.UG_Tier <= 3)
              ? parsedResumeData.UG_Tier
              : (parsedResumeData.UG_InstituteName ? 3 : null), // Default to tier 3 if institution exists but tier not provided, null if empty
            pg_tier: (parsedResumeData.PG_Tier !== undefined && parsedResumeData.PG_Tier !== null && parsedResumeData.PG_Tier >= 1 && parsedResumeData.PG_Tier <= 3)
              ? parsedResumeData.PG_Tier
              : (parsedResumeData.PG_InstituteName ? 3 : null), // Default to tier 3 if institution exists but tier not provided, null if empty
            phd_institute: parsedResumeData.PhD_Institute || 0,
            longevity_years: parsedResumeData.Longevity_Years || 0,
            number_of_jobs: parsedResumeData.No_of_Jobs || 0,
            average_experience: parsedResumeData.Experience_Average || 0,
            skills_count: parsedResumeData.Skills_No || 0,
            achievements_count: parsedResumeData.Achievements_No || 0,
            achievements: parsedResumeData.Achievements || [],
            trainings_count: parsedResumeData.Trainings_No || 0,
            trainings: parsedResumeData.Trainings || [],
            workshops_count: parsedResumeData.Workshops_No || 0,
            workshops: parsedResumeData.Workshops || [],
            research_papers_count: parsedResumeData.Research_Papers_No || 0,
            research_papers: parsedResumeData.Research_Papers || [],
            patents_count: parsedResumeData.Patents_No || 0,
            patents: parsedResumeData.Patents || [],
            books_count: parsedResumeData.Books_No || 0,
            books: parsedResumeData.Books || [],
            is_jk: parsedResumeData.State_JK || 0,
            projects_count: parsedResumeData.Projects_No || 0,
            projects: parsedResumeData.Projects || [],
            best_fit_for: parsedResumeData.Best_Fit_For || "",
            profile_score: parsedResumeData.Profile_Score || 0,
            fitment_score: parsedResumeData.Fitment_Score || 0
          };

          // Ensure we have an email - try to extract from raw content if not found
          if (!mappedData.personalInfo.email || mappedData.personalInfo.email.trim() === "") {
            const extractedEmail = extractEmailFromText(parsedContent);
            if (extractedEmail) {
              mappedData.personalInfo.email = extractedEmail;
            }
          }

          // Add job_id to the mapped data
          mappedData.job_id = selectedJob;

          // Store resume data in your preferred storage
          const resumeData = {
            user_id: 'your_user_id',
            file_name: file.name,
            file_type: file.type,
            file_size: file.size,
            parsed_data: mappedData,
            status: 'processed',
            created_at: new Date().toISOString(),
            job_id: selectedJob, // Store selected job ID
          };

          console.log('Resume data processed:', resumeData);

          // Do not send invitation here; it will be sent after Save Resume succeeds

          // Clear progress interval and set to 100%
          clearInterval(interval);
          setProgress(100);

          // Notify user and return data
          setTimeout(() => {
            // Implement your own toast notification here
            onResumeUploaded(mappedData, file);
            setUploading(false);
            onParsingStateChange(false);
          }, 500);
        } catch (jsonError) {
          console.error("Error parsing JSON:", jsonError);
          console.error("Raw JSON text:", jsonText);
          throw new Error("Gemini returned an invalid JSON payload while parsing the resume.");
        }
      } catch (error: any) {
        clearInterval(interval);
        const message = error?.message || 'Failed to process resume due to an unknown error.';
        setError(message);
        setUploading(false);
        onParsingStateChange(false);
        console.error("Error parsing resume:", error);
      }
    } catch (error: any) {
      const message = error?.message || 'Failed to process resume due to an unexpected error.';
      setError(message);
      setUploading(false);
      onParsingStateChange(false);
      console.error("Error parsing resume (outer catch):", error);
    }
  };

  const readFileAsBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          // Extract base64 data without the prefix
          const base64 = reader.result.split(',')[1];
          resolve(base64);
        } else {
          reject(new Error('Failed to read file as base64'));
        }
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  // Function to attempt to send personality test invite using various email extraction methods
  const sendPersonalityTestInvite = async (parsedResumeData: ResumeData) => {
    try {
      // First try to use the email from the parsed data
      let candidateEmail = parsedResumeData.personalInfo.email;

      // If no email was found, try to extract from other fields
      if (!candidateEmail) {
        // Try from summary
        if (parsedResumeData.personalInfo.summary) {
          const extractedEmail = extractEmailFromText(parsedResumeData.personalInfo.summary);
          if (extractedEmail) candidateEmail = extractedEmail;
        }

        // If still no email, try from other text fields
        if (!candidateEmail) {
          const allText = JSON.stringify(parsedResumeData);
          const extractedEmail = extractEmailFromText(allText);
          if (extractedEmail) candidateEmail = extractedEmail;
        }
      }

      if (!candidateEmail) {
        console.error("No email could be extracted from the resume");
        return;
      }

      // Fetch resume data from your preferred storage
      // Replace this with your own data fetching logic
      console.log('Would fetch resume data for user:', 'your_user_id');

      // Mock data for demonstration
      const data = {
        parsed_data: {
          personalInfo: {
            email: 'user@example.com',
            name: 'User Name'
          }
        }
      };

      if (!data?.parsed_data) {
        throw new Error('No resume data found');
      }

      // Handle personality test invite
      console.log('Would send personality test invite to:', data.parsed_data.personalInfo?.email);
      // Implement your own invitation logic here
    } catch (error) {
      console.error('Failed to send personality test invite:', error);
    }
  };

  return (
    <div className="flex flex-col items-center space-y-4">
      {/* Job Selection Dropdown */}
      <div className="w-full">
        <label className="text-sm font-medium mb-2 block text-gray-700">
          Select Job Position <span className="text-red-500">*</span>
        </label>
        {loadingJobs ? (
          <div className="text-sm text-gray-500">Loading jobs...</div>
        ) : (
          <Select value={selectedJob} onValueChange={setSelectedJob}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select a job position for this resume" />
            </SelectTrigger>
            <SelectContent>
              {jobs.length === 0 ? (
                <SelectItem value="" disabled>
                  No active jobs available
                </SelectItem>
              ) : (
                jobs.map((job) => (
                  <SelectItem key={job.id} value={job.id}>
                    {job.title}
                  </SelectItem>
                ))
              )}
            </SelectContent>
          </Select>
        )}
        {jobs.length === 0 && !loadingJobs && (
          <p className="text-sm text-gray-500 mt-2">
            Please create an active job in the Jobs page first.
          </p>
        )}
      </div>

      <div
        className={`border-2 border-dashed rounded-lg p-10 w-full text-center cursor-pointer transition-colors
          ${dragging ? 'bg-purple-50 border-purple-400' : 'bg-gray-50 border-gray-300'}
          ${!selectedJob ? 'opacity-50 cursor-not-allowed' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => {
          if (selectedJob) {
            document.getElementById('file-upload')?.click();
          } else {
            toast.error("Please select a job first");
          }
        }}
      >
        <input
          id="file-upload"
          type="file"
          className="hidden"
          accept=".pdf,.doc,.docx"
          onChange={handleFileChange}
        />
        <div className="flex flex-col items-center gap-4">
          <div className="w-16 h-16 flex items-center justify-center bg-purple-50 rounded-full">
            {getFileIcon()}
          </div>

          <div>
            {!file && (
              <>
                <p className="text-lg font-medium mb-2">
                  {selectedJob ? "Drop your resume here or click to browse" : "Please select a job first"}
                </p>
                <p className="text-sm text-gray-500">Supports PDF, DOC, DOCX (Max 10MB)</p>
              </>
            )}

            {file && (
              <div className="mt-2">
                <p className="text-lg font-medium">{file.name}</p>
                <p className="text-sm text-gray-500">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {error && (
        <Alert variant="destructive" className="mt-4 w-full">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {file && !uploading && (
        <div className="mt-6 flex gap-4">
          <Button
            variant="outline"
            onClick={() => { setFile(null); setError(null); }}
          >
            Choose Different File
          </Button>
          <Button
            onClick={parseResume}
            className="bg-purple-600 hover:bg-purple-700 text-white"
            disabled={!selectedJob}
          >
            Parse Resume
          </Button>
        </div>
      )}

      {uploading && (
        <div className="mt-6 w-full">
          <div className="flex justify-between mb-2">
            <span className="text-sm font-medium">Parsing resume...</span>
            <span className="text-sm font-medium">{progress}%</span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>
      )}
    </div>
  );
}
