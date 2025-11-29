export interface ResumeData {
  personalInfo: {
    name: string;
    email: string;
    phone: string;
    address?: string;
    summary?: string;
  };
  education: Education[];
  experience: Experience[];
  skills: string[];
  ug_institute?: string;
  ug_tier?: number | null; // 1, 2, 3, or null
  pg_institute?: string;
  pg_tier?: number | null; // 1, 2, 3, or null
  phd_institute: number; // 0 for no, 1 for yes
  longevity_years: number; // working years count
  number_of_jobs: number;
  average_experience: number; // longevity/number of jobs
  skills_count: number;
  achievements_count: number;
  achievements: string[];
  trainings_count: number;
  trainings: string[];
  workshops_count: number;
  workshops: string[];
  research_papers: string[];
  research_papers_count: number;
  patents: string[];
  patents_count: number;
  books: string[];
  books_count: number;
  is_jk: number; // 0 for no, 1 for yes (J&K)
  projects_count: number;
  projects: string[];
  best_fit_for?: string; // Making this optional and adding it to the interface
  profile_score?: number;
  fitment_score?: number;
  job_id?: string; // ID of the job position this resume is being evaluated for
}

export interface Education {
  institution: string;
  degree: string;
  field?: string;
  startDate?: string;
  endDate?: string;
  gpa?: string;
}

export interface Experience {
  company: string;
  position: string;
  startDate?: string;
  endDate?: string;
  description?: string;
  location?: string;
}
