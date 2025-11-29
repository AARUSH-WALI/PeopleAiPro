import { useState, useEffect } from "react";
import { Briefcase, Calendar, Users, FileText, Loader2, AlertCircle } from "lucide-react";
import axios from "axios";
import { format } from "date-fns";
import { toast } from "sonner";
import StatCard from "@/components/dashboard/StatCard";
import JobFitmentTable from "@/components/dashboard/JobFitmentTable";
import CandidateScores from "@/components/dashboard/CandidateScores";
import InterviewScheduleDialog from "@/components/dashboard/InterviewScheduleDialog";
// import RecentActivity from "@/components/dashboard/RecentActivity";
// import CandidateStatus from "@/components/dashboard/CandidateStatus";
import { DashboardStats, Candidate, Interview } from "@/types/dashboard";

// Use production API URL in production environment, otherwise use development URL
const API_BASE_URL = import.meta.env.PROD
  ? import.meta.env.VITE_PROD_API_BASE_URL
  : import.meta.env.VITE_DEV_API_BASE_URL || 'http://localhost:5000/api';

const vacancyData = [
  { name: "Jan", value: 12 },
  { name: "Feb", value: 19 },
  { name: "Mar", value: 15 },
  { name: "Apr", value: 23 },
  { name: "May", value: 18 },
  { name: "Jun", value: 15 },
  { name: "Jul", value: 20 },
  { name: "Aug", value: 25 },
  { name: "Sep", value: 15 },
  { name: "Oct", value: 10 },
  { name: "Nov", value: 15 },
  { name: "Dec", value: 20 },
];

const candidates = [
  {
    name: "Saksham Gupta",
    email: "2022a6041@mietjammu.in",
    fitmentScore: 65.5,
    phone: "+91 9876543210",
    education: "Ph.D. in Computer Science",
    experience: "3+ Years",
    skills: ["React", "TypeScript", "UI/UX Design"],
    projects: ["AI-based Recommendation System", "E-Learning Platform"],
    expectedRole: "Assistant Professor",
    location: "Jammu",
    longevityScore: 85
  },
  {
    name: "Ayush Thakur",
    email: "ayushthakur1412@gmail.com",
    fitment_score: 69.94,
    phone: "+91 9876543211",
    education: "Master of Computer Science",
    experience: "6+ Years",
    skills: ["Web Development", "JavaScript", "Node.js"],
    projects: ["E-Commerce Platform", "Hospital Management System"],
    expected_role: "Professor",
    location: "Delhi",
    longevity_score: 78
  },
  {
    name: "Adishwar Sharma",
    email: "2021a1045@mietjammu.in",
    fitment_score: 72.58,
    phone: "+91 9876543212",
    education: "Ph.D. in AI",
    experience: "4+ Years",
    skills: ["Machine Learning", "Data Analysis", "Python"],
    projects: ["Predictive Analytics Tool", "Natural Language Processing System"],
    expectedRole: "Research Associate",
    location: "Bangalore",
    longevityScore: 92
  },
  {
    name: "Garima Saigal",
    email: "garimasaigal02@gmail.com",
    fitmentScore: 55.32,
    phone: "+91 9876543213",
    education: "M.Tech in Software Engineering",
    experience: "2+ Years",
    skills: ["Java", "Spring Boot", "SQL"],
    projects: ["Banking Application", "Inventory Management"],
    expectedRole: "Lab Assistant",
    location: "Mumbai",
    longevityScore: 65
  },
  {
    name: "Aarush Wali",
    email: "2022A6002@mietjammu.in",
    fitmentScore: 62.45,
    phone: "+91 9876543214",
    education: "M.Sc. in AI",
    experience: "2+ Years",
    skills: ["Computer Vision", "NLP", "TensorFlow"],
    projects: ["Facial Recognition System", "Chatbot Implementation"],
    expectedRole: "AI Researcher",
    location: "Hyderabad",
    longevityScore: 80
  }
];

const jobRoles = [
  "Professor",
  "Assistant Professor",
  "Associate Professor",
  "Head of Department",
  "Dean",
  "Research Associate",
  "Lab Assistant",
  "Academic Coordinator"
];

const fitCategories = ["Best Fit", "Mid Fit", "Not Fit"];

const mockEmployeeData = [
  { name: "Gandharv Kaloo", role: "Professor", fitment: "Best Fit" },
  { name: "Saksham Gupta", role: "Assistant Professor", fitment: "Best Fit" },
  { name: "Aarush Wali", role: "Associate Professor", fitment: "Mid Fit" },
  { name: "Abhishek Kumar", role: "Professor", fitment: "Not Fit" },
  { name: "Dhruv Gupta", role: "Dean", fitment: "Mid Fit" },
  { name: "Antra Bali", role: "Research Associate", fitment: "Best Fit" },
  { name: "Karan Patel", role: "Lab Assistant", fitment: "Mid Fit" },
  { name: "Rohit Sharma", role: "Academic Coordinator", fitment: "Best Fit" },
  { name: "Ansh Patyal", role: "Professor", fitment: "Mid Fit" },
  { name: "Naman Kumar", role: "Associate Professor", fitment: "Mid Fit" },
  { name: "Archit Singh", role: "Assistant Professor", fitment: "Mid Fit" },
  { name: "Aditya Raina", role: "Professor", fitment: "Best Fit" },
  { name: "Sameer Rizvi", role: "Professor", fitment: "Best Fit" },
  { name: "Sachin Sharma", role: "Professor", fitment: "Mid Fit" },
  { name: "Vansh Kapoor", role: "Assistant Professor", fitment: "Not Fit" },
  { name: "Mohammad Kaif", role: "Professor", fitment: "Best Fit" },
  { name: "Ajay Kumar", role: "Associate Professor", fitment: "Mid Fit" },
  { name: "Gurdeep Singh", role: "Professor", fitment: "Best Fit" },
  { name: "Pankaj Jandiyal", role: "Associate Professor", fitment: "Not Fit" },
  { name: "Akshit Kumar ", role: "Associate Professor", fitment: "Mid Fit" },
  { name: "Pankaj Singh", role: "Professor", fitment: "Not Fit" },
  { name: "Saksham Bamotra", role: "Head of Department", fitment: "Best Fit" },
  { name: "Amit Kumar", role: "Head of Department", fitment: "Mid Fit" },
  { name: "Ajinkey Rahane", role: "Head of Department", fitment: "Not Fit" },
  { name: "Rameshwar Prasad", role: "Professor", fitment: "Mid Fit" },
  { name: "Aryan Pandoh", role: "Dean", fitment: "Best Fit" },
  { name: "Aryan Gupta", role: "Dean", fitment: "Not Fit" },
  { name: "Punya Sharma", role: "Research Associate", fitment: "Mid Fit" },
  { name: "Aakarsh Gupta", role: "Research Associate", fitment: "Not Fit" },
  { name: "Parth Gupta", role: "Dean", fitment: "Mid Fit" },
  { name: "Preetika Sharma", role: "Lab Assistant", fitment: "Best Fit" },
  { name: "Veena Rajput", role: "Lab Assistant", fitment: "Not Fit" },
  { name: "Punya Sharma", role: "Research Associate", fitment: "Mid Fit" },
  { name: "Venika Sharma", role: "Academic Coordinator", fitment: "Best Fit" },
  { name: "Aditya Nanda", role: "Academic Coordinator", fitment: "Mid Fit" },
  { name: "Ajit Singh", role: "Academic Coordinator", fitment: "Not Fit" },
];

export default function Dashboard() {
  const [showInterviews, setShowInterviews] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<DashboardStats>({
    totalApplications: 0,
    hiredCandidates: 0,
    pendingResumes: 0,
    todayInterviews: 0
  });
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [interviews, setInterviews] = useState<Interview[]>([]);
  const [jobRoles, setJobRoles] = useState<string[]>([]);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        setError('');

        // Fetch all data in parallel with error handling for each request
        const [resumesRes, interviewsRes, jobsRes] = await Promise.all([
          axios.get(`${API_BASE_URL}/resumes`, {
            params: {
              limit: 100, // Get more resumes to have enough data for the dashboard
              sort_by: 'created_at',
              sort_order: 'desc'
            }
          }).catch(err => {
            console.error('Error fetching resumes:', err);
            return { data: { data: [], pagination: { total: 0 } } };
          }),
          axios.get(`${API_BASE_URL}/interviews`, {
            params: {
              start_date: new Date().toISOString().split('T')[0],
              end_date: new Date().toISOString().split('T')[0],
              limit: 10,
              sort_by: 'date',
              sort_order: 'asc'
            }
          }).catch(err => {
            console.error('Error fetching interviews:', err);
            return { data: { data: [] } };
          }),
          axios.get(`${API_BASE_URL}/jobs`, {
            params: {
              limit: 50,
              fields: 'title,department,required_skills',
              status: 'active'
            }
          }).catch(err => {
            console.error('Error fetching jobs:', err);
            return { data: { data: [] } };
          })
        ]);

        // Get resumes data with error handling
        const resumesData = resumesRes.data?.data || [];
        const totalApplications = resumesRes.data?.pagination?.total || 0;
        const todayInterviews = interviewsRes.data?.data?.length || 0;
        const jobsData = jobsRes.data?.data || [];

        // Calculate statistics
        const hiredCandidates = resumesData.filter(
          (r: any) => r.status === 'hired' || r.status === 'accepted'
        ).length;

        const pendingResumes = resumesData.filter(
          (r: any) => !r.status || ['pending', 'review', 'new'].includes(r.status)
        ).length;

        // Update stats
        setStats({
          totalApplications,
          hiredCandidates,
          pendingResumes,
          todayInterviews
        });

        // Notify on new candidates since last visit
        try {
          const key = 'lastResumeCount';
          const prev = Number(localStorage.getItem(key) || '0');
          if (!Number.isNaN(prev) && totalApplications > prev) {
            const delta = totalApplications - prev;
            if (delta > 0) {
              toast.success(`${delta} new candidate${delta > 1 ? 's' : ''} registered since your last visit`);
            }
          }
          localStorage.setItem(key, String(totalApplications));
        } catch { }

        // Process candidates for the CandidateScores component
        const formattedCandidates = resumesData.map((resume: any) => {
          // Use fitment_score from API if available, otherwise use 0
          // The API should calculate this on the backend based on resume analysis
          const fitmentScore = typeof resume.fitment_score === 'number'
            ? Math.min(100, Math.max(0, resume.fitment_score))
            : 0; // Default to 0 if not provided by API

          return {
            id: resume.id || '',
            name: resume.name || resume.candidate_name || 'Unknown Candidate',
            email: resume.email || resume.candidate_email || '',
            fitment_score: fitmentScore,
            phone: resume.phone || '',
            education: resume.education || '',
            experience: resume.experience ? `${resume.experience} years` : 'Not specified',
            skills: Array.isArray(resume.skills) ? resume.skills : [],
            projects: Array.isArray(resume.projects) ? resume.projects : [],
            expectedRole: resume.expected_role || resume.best_fit_for || 'Not specified',
            location: resume.location || 'Location not specified',
            longevity_score: resume.longevity_years || 0,
            status: resume.status || 'pending'
          };
        });

        setCandidates(formattedCandidates);

        // Process job roles from jobs data
        const jobRoles = Array.from(new Set(
          jobsData
            .map((job: any) => job.title || job.role || job.department)
            .filter(Boolean)
        )).slice(0, 10); // Limit to 10 roles max

        // If no job roles from jobs, try to get from resumes
        if (jobRoles.length === 0) {
          const resumeJobRoles = Array.from(new Set(
            resumesData
              .map((r: any) => r.expected_role || r.best_fit_for)
              .filter(Boolean)
          )).slice(0, 10);

          setJobRoles(resumeJobRoles as string[]);
        } else {
          setJobRoles(jobRoles as string[]);
        }

        // Process interviews for the interview dialog
        const interviewsData = interviewsRes.data?.data || [];
        const formattedInterviews = interviewsData.map((i: any) => ({
          id: i.id || '',
          title: i.title || 'Interview',
          candidateName: i.candidate_name || 'Unknown',
          candidateEmail: i.candidate_email || '',
          date: i.date || new Date().toISOString(),
          status: i.status || 'scheduled',
          interviewType: i.interview_type || '',
          meetingPlatform: i.meeting_platform || '',
          description: i.description || ''
        }));

        setInterviews(formattedInterviews);

        // Process interviews
        if (interviewsRes.data?.data) {
          const interviewsData = Array.isArray(interviewsRes.data.data)
            ? interviewsRes.data.data
            : [interviewsRes.data.data];

          const formattedInterviews = interviewsData.map((i: any) => ({
            id: i.id || '',
            title: i.title || 'Interview',
            candidateName: i.candidate_name || 'Unknown',
            candidateEmail: i.candidate_email || '',
            date: i.date || new Date().toISOString(),
            status: i.status || 'scheduled',
            interviewType: i.interview_type || '',
            meetingPlatform: i.meeting_platform || '',
            description: i.description || ''
          }));

          setInterviews(formattedInterviews);
        }

        // Set default job roles (can be updated later if needed)
        setJobRoles([
          'Software Engineer',
          'Product Manager',
          'UX Designer',
          'Data Scientist',
          'DevOps Engineer'
        ]);

      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard data. Please try again later.');
        toast.error('Failed to load dashboard data');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  if (loading) {
    return (
      <div className="page-container flex items-center justify-center">
        <Loader2 className="h-12 w-12 animate-spin text-purple-600" />
      </div>
    );
  }


  if (error) {
    return (
      <div className="page-container flex flex-col items-center justify-center text-red-600">
        <AlertCircle className="h-12 w-12 mb-4" />
        <p className="text-lg">{error}</p>
      </div>
    );
  }

  return (
    <div className="page-container bg-white text-gray-800">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <StatCard
          value={stats.totalApplications.toString()}
          title="Job Applications"
          icon={<Users size={40} className="text-white" />}
          color="purple"
          className="shadow-md"
        />

        <StatCard
          value={stats.hiredCandidates.toString()}
          title="Hired Candidates"
          icon={<Briefcase size={40} className="text-white" />}
          color="blue"
          className="relative overflow-hidden shadow-md"
        />

        <StatCard
          value={stats.pendingResumes.toString()}
          title="Resumes for Review"
          icon={<FileText size={40} className="text-white" />}
          color="green"
          className="shadow-md"
        />

        <StatCard
          value={stats.todayInterviews.toString()}
          title="Scheduled Interviews For Today"
          icon={<Calendar size={40} className="text-blue" />}
          color="rose"
          onClick={() => setShowInterviews(true)}
          className="cursor-pointer hover:bg-gray-100 transition-colors shadow-md"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="bg-white shadow-md rounded-lg overflow-hidden border border-gray-200 h-full">
          <JobFitmentTable
            jobRoles={jobRoles}
            fitCategories={['Best Fit', 'Mid Fit', 'Not Fit']}
            employees={candidates.map(c => ({
              name: c.name,
              role: c.expectedRole || 'N/A',
              fitment: c.fitment_score >= 70 ? 'Best Fit' : c.fitment_score >= 40 ? 'Mid Fit' : 'Not Fit'
            }))}
          />
        </div>

        <div className="bg-white shadow-md rounded-lg overflow-hidden border border-gray-200 h-full">
          <CandidateScores candidates={candidates} />
        </div>
      </div>

      {/* <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <RecentActivity />
        <CandidateStatus approvedCount={45} reviewCount={30} rejectedCount={25} />
      </div> */}

      <InterviewScheduleDialog
        isOpen={showInterviews}
        onClose={() => setShowInterviews(false)}
        interviews={interviews}
        isLoading={loading}
      />
    </div>
  );
}
