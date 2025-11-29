export interface DashboardStats {
  totalApplications: number;
  hiredCandidates: number;
  pendingResumes: number;
  todayInterviews: number;
}

export interface Candidate {
  id: string;
  name: string;
  email: string;
  fitment_score: number;
  phone: string;
  education: string;
  experience: string;
  skills: string[];
  projects: string[];
  expectedRole: string;
  location: string;
  longevity_score: number;
  status: 'hired' | 'rejected' | 'pending';
}

export interface Interview {
  id: string;
  title: string;
  candidateName: string;
  candidateEmail: string;
  date: string;
  status: 'scheduled' | 'completed' | 'cancelled';
  interviewType: string;
  meetingPlatform: string;
  candidateId?: string;
  description?: string;
  interviewerName?: string;
  interviewerEmail?: string;
  meetingLink?: string;
}
