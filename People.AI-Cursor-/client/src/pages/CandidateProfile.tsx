import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import FitmentScoreGauge from "@/components/users/FitmentScoreGauge";
import PersonalityPieChart from "@/components/users/PersonalityPieChart";
import RetentionAnalysis from "@/components/users/RetentionAnalysis";
import type { UserProfile } from "./Users";

const API_BASE_URL = import.meta.env.PROD
    ? import.meta.env.VITE_PROD_API_BASE_URL
    : import.meta.env.VITE_DEV_API_BASE_URL || 'http://localhost:5000/api';

const mapResumeToUserProfile = (resume: any): UserProfile => ({
    id: resume.id || '',
    name: resume.name || 'Unknown',
    email: resume.email || '',
    score: resume.fitment_score || 0,
    jobRole: resume.best_fit_for || 'Not Specified',
    experience: typeof resume.longevity_years === 'number' ? `${resume.longevity_years} years` : 'No experience',
    education: Array.isArray(resume.education) && resume.education[0]
        ? `${resume.education[0].degree} in ${resume.education[0].field || ''} from ${resume.education[0].institution || ''}`.trim()
        : '',
    about: resume.summary || 'No summary available',
    profileImage: '',
    personalityScores: {
        extraversion: resume.extraversion || 0,
        agreeableness: resume.agreeableness || 0,
        openness: resume.openness || 0,
        neuroticism: resume.neuroticism || 0,
        conscientiousness: resume.conscientiousness || 0
    },
    status: resume.status || 'pending',
    phone: resume.phone || '',
    address: resume.address || '',
    summary: resume.summary || '',
    best_fit_for: resume.best_fit_for || '',
    created_at: resume.created_at || new Date().toISOString(),
    fitment_score: resume.fitment_score || 0,
    skills: Array.isArray(resume.skills) ? resume.skills : [],
    candidate_type: resume.candidate_type,
    file_url: resume.file_url,
    resume_url: resume.resume_url || resume.file_url,
    personality_score: resume.personality_score || undefined,
    dataset_score: resume.dataset_score || undefined,
    retention_score: resume.retention_score || undefined,
    retention_risk: resume.retention_risk || undefined,
    retention_analysis: resume.retention_analysis || undefined,
});


export default function CandidateProfile() {
    const { id } = useParams();
    const navigate = useNavigate();
    const [user, setUser] = useState<UserProfile | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [retentionInsight, setRetentionInsight] = useState<string | null>(null);

    useEffect(() => {
        const loadCandidateData = async () => {
            if (!id) {
                setError('Invalid candidate ID');
                setLoading(false);
                return;
            }

            try {
                setLoading(true);
                setError(null);

                console.log(`[CandidateProfile] Fetching resume data for ID: ${id}`);

                // Fetch resume data from API (which queries Supabase)
                const response = await axios.get(`${API_BASE_URL}/resumes/${id}`);

                // Handle response structure
                const resume = response.data?.data || response.data;

                if (!resume) {
                    setError('Resume not found');
                    setLoading(false);
                    return;
                }

                console.log('[CandidateProfile] Resume data received:', {
                    id: resume.id,
                    name: resume.name,
                    email: resume.email
                });

                // Map resume to user profile
                setUser(mapResumeToUserProfile(resume));

                if (resume.retention_score || resume.retention_risk || resume.retention_analysis) {
                    try {
                        const insightResponse = await axios.get(`${API_BASE_URL}/resumes/${id}/retention-insights`);
                        const insightText = insightResponse.data?.insight || insightResponse.data?.data?.insight;
                        if (insightText) {
                            setRetentionInsight(insightText);
                        }
                    } catch (insightError: any) {
                        console.error('[CandidateProfile] Error loading retention insights:', insightError);
                    }
                }

            } catch (err: any) {
                console.error('[CandidateProfile] Error loading candidate:', err);
                const errorMessage = err.response?.data?.error
                    || err.response?.data?.message
                    || err.message
                    || 'Failed to load candidate data';
                setError(errorMessage);
            } finally {
                setLoading(false);
            }
        };

        loadCandidateData();
    }, [id]);

    if (loading) {
        return (
            <div className="page-container flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600 mx-auto mb-4"></div>
                    <p className="text-gray-600">Loading candidate profile...</p>
                </div>
            </div>
        );
    }

    if (error || !user) {
        return (
            <div className="page-container">
                <div className="p-4">
                    <Button variant="ghost" onClick={() => navigate(-1)}>Back</Button>
                </div>
                <div className="flex items-center justify-center h-[calc(100vh-8rem)]">
                    <div className="text-center">
                        <p className="text-red-600 text-lg font-semibold mb-2">{error || 'Candidate not found'}</p>
                        <Button onClick={() => navigate(-1)}>Go Back</Button>
                    </div>
                </div>
            </div>
        );
    }

    const openResume = () => {
        // Prefer resume_url (Supabase Storage public URL) over file_url
        const resumeUrl = (user as any).resume_url || user.file_url;

        if (resumeUrl) {
            // If it's already a full URL (starts with http), use it directly
            // Otherwise, it might be a relative path
            const url = resumeUrl.startsWith('http')
                ? resumeUrl
                : resumeUrl.startsWith('/')
                    ? `${window.location.origin}${resumeUrl}`
                    : `${window.location.origin}/${resumeUrl}`;
            window.open(url, '_blank', 'noopener,noreferrer');
        } else {
            console.warn('No resume URL available for candidate:', user.id);
        }
    };

    return (
        <div className="page-container">
            <div className="p-4">
                <Button variant="ghost" onClick={() => navigate(-1)}>Back</Button>
            </div>
            <ScrollArea className="h-[calc(100vh-6rem)] w-full">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-6">
                    {/* Left Column - Candidate Info */}
                    <div className="space-y-6">
                        <div className="flex flex-col md:flex-row gap-6 items-center md:items-start">
                            <div className="aspect-square w-full max-w-[250px] bg-gray-100 rounded-lg flex items-center justify-center">
                                {user.profileImage ? (
                                    <img src={user.profileImage} alt="Profile" className="w-full h-full object-cover rounded-lg" />
                                ) : (
                                    <span className="text-gray-400">Profile Image</span>
                                )}
                            </div>
                            <div className="space-y-3 flex-1">
                                <h3 className="text-3xl font-semibold">{user.name}</h3>
                                <p className="text-gray-500 text-lg">{user.email}</p>
                                <div className="pt-2">
                                    <div className="inline-block bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm font-medium">
                                        {user.jobRole}
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="space-y-4 bg-gray-50 p-4 rounded-lg">
                            <div>
                                <h4 className="font-medium text-lg">Experience:</h4>
                                <p className="text-base">{user.experience}</p>
                            </div>
                            <div>
                                <h4 className="font-medium text-lg">Education:</h4>
                                <p className="text-base">{user.education || 'Not specified'}</p>
                            </div>
                            <div>
                                <h4 className="font-medium text-lg">About Me:</h4>
                                <p className="text-base text-gray-600">{user.about}</p>
                            </div>
                            <Button
                                className="w-full"
                                onClick={openResume}
                                disabled={!((user as any).resume_url || user.file_url)}
                            >
                                {((user as any).resume_url || user.file_url) ? 'View Resume' : 'No Resume Available'}
                            </Button>
                        </div>
                    </div>

                    {/* Right Column - Scores and Analysis */}
                    <div className="space-y-6">
                        {/* Fitment Score */}
                        <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                            <h3 className="text-2xl font-semibold mb-2 text-center">AI Generated Fitment Score</h3>
                            <div className="w-full max-w-[220px] mx-auto">
                                <FitmentScoreGauge score={user.score} />
                            </div>
                            <div className="mt-4">
                                <h4 className="font-medium text-center mb-1 text-lg">Score Breakdown</h4>
                                {(() => {
                                    const overall = user.fitment_score ?? user.score ?? 0;
                                    const datasetScore = user.dataset_score; // Raw dataset score (0-100) - stored in DB
                                    
                                    // Determine candidate type similar to backend normalization
                                    const rawType = typeof user.candidate_type === 'string' ? user.candidate_type.trim().toLowerCase() : '';
                                    const isExperienced = rawType === 'experienced';
                                    
                                    // If dataset_score is available, use actual scaled values
                                    if (datasetScore !== undefined && datasetScore !== null && user.personality_score !== undefined) {
                                        // Calculate scaled dataset score based on category
                                        // Experienced: dataset 70%, Big5 30%
                                        // Fresher/Intermediate: dataset 30%, Big5 70%
                                        const datasetScaled = isExperienced 
                                            ? (datasetScore / 100) * 70  // Experienced: 70% dataset
                                            : (datasetScore / 100) * 30; // Fresher/Intermediate: 30% dataset
                                        
                                        // personality_score is already the scaled Big5 score
                                        const big5Scaled = user.personality_score;
                                        
                                        return (
                                            <div className="space-y-1 px-5">
                                                <div className="flex justify-between text-sm">
                                                    <span className="font-medium">Resume Fit (Profile & Experience):</span>
                                                    <span>{datasetScaled.toFixed(1)}</span>
                                                </div>
                                                <div className="flex justify-between text-sm">
                                                    <span className="font-medium">Personality Fit (Big5):</span>
                                                    <span>{big5Scaled.toFixed(1)}</span>
                                                </div>
                                            </div>
                                        );
                                    } else {
                                        // Fallback: calculate from overall score if dataset_score not available
                                        const datasetShare = isExperienced ? 70 : 30;
                                        const big5Share = 100 - datasetShare;
                                        const datasetPoints = (overall * datasetShare) / 100;
                                        const big5Points = (overall * big5Share) / 100;

                                        return (
                                            <div className="space-y-1 px-5">
                                                <div className="flex justify-between text-sm">
                                                    <span className="font-medium">Resume Fit (Profile & Experience):</span>
                                                    <span>{datasetPoints.toFixed(1)}</span>
                                                </div>
                                                <div className="flex justify-between text-sm">
                                                    <span className="font-medium">Personality Fit (Big5):</span>
                                                    <span>{big5Points.toFixed(1)}</span>
                                                </div>
                                            </div>
                                        );
                                    }
                                })()}
                            </div>
                        </div>

                        {/* Personality Assessment */}
                        {user.personalityScores && (
                            <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm space-y-4">
                                <h4 className="text-2xl font-semibold text-center">Personality Assessment</h4>
                                <div className="w-full h-[300px] max-w-[420px] mx-auto">
                                    <PersonalityPieChart scores={user.personalityScores} />
                                </div>
                            </div>
                        )}

                        {/* Retention Analysis */}
                        {user.retention_score !== undefined && (
                            <RetentionAnalysis 
                                retentionData={{
                                    retention_score: user.retention_score,
                                    retention_risk: user.retention_risk,
                                    retention_analysis: user.retention_analysis
                                }}
                                retentionInsight={retentionInsight}
                            />
                        )}
                    </div>
                </div>
            </ScrollArea>
        </div>
    );
}
