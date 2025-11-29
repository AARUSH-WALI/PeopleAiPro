import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ChevronLeft } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { UserProfile } from "@/pages/Users";
import PersonalityPieChart from "./PersonalityPieChart";
import FitmentScoreGauge from "./FitmentScoreGauge";

import { format } from "date-fns";
import { toast } from "@/components/ui/use-toast";
import { useEffect, useState } from "react";
import axios from "axios";

const API_BASE_URL = import.meta.env.PROD
  ? import.meta.env.VITE_PROD_API_BASE_URL
  : import.meta.env.VITE_DEV_API_BASE_URL || 'http://localhost:5000/api';

interface UserDetailsDialogProps {
  isOpen: boolean;
  onClose: () => void;
  user: UserProfile | null;
}

export default function UserDetailsDialog({ isOpen, onClose, user }: UserDetailsDialogProps) {
  const [resumeUrl, setResumeUrl] = useState<string | null>(null);
  const [retentionData, setRetentionData] = useState<{
    retention_score?: number | null;
    retention_risk?: string | null;
    retention_analysis?: any;
  } | null>(null);
  const [retentionInsight, setRetentionInsight] = useState<string | null>(null);
  const [loadingInsight, setLoadingInsight] = useState(false);

  useEffect(() => {
    const loadResumeData = async () => {
      if (!user?.id || !isOpen) {
        setResumeUrl(null);
        setRetentionData(null);
        return;
      }

      try {
        console.log('[UserDetailsDialog] Loading resume data for user:', user.id);
        const response = await axios.get(`${API_BASE_URL}/resumes/${user.id}`);
        const resume = response.data?.data || response.data;

        console.log('[UserDetailsDialog] Resume data loaded:', {
          id: resume?.id,
          resume_url: resume?.resume_url,
          file_url: resume?.file_url,
          retention_score: resume?.retention_score,
          retention_risk: resume?.retention_risk
        });

        if (resume) {
          // Extract resume URL (prefer resume_url over file_url)
          const url = resume.resume_url || resume.file_url || null;
          setResumeUrl(url);
          console.log('[UserDetailsDialog] Resume URL set:', url);

          // Extract retention data
          if (resume.retention_score !== undefined || resume.retention_risk) {
            setRetentionData({
              retention_score: resume.retention_score ?? null,
              retention_risk: resume.retention_risk ?? null,
              retention_analysis: resume.retention_analysis ?? null
            });

            // Fetch AI-generated retention insight
            fetchRetentionInsight(user.id);
          }
        }
      } catch (error) {
        console.error('[UserDetailsDialog] Error loading resume data:', error);
        setResumeUrl(null);
        setRetentionData(null);
        setRetentionInsight(null);
      }
    };

    const fetchRetentionInsight = async (resumeId: string) => {
      try {
        setLoadingInsight(true);
        setRetentionInsight(null); // Clear old insight first
        console.log('[UserDetailsDialog] Fetching retention insight for:', resumeId);
        const response = await axios.get(`${API_BASE_URL}/resumes/${resumeId}/retention-insights`, {
          params: {
            _t: Date.now() // Add timestamp to prevent caching
          }
        });
        console.log('[UserDetailsDialog] Retention insight response:', response.data);
        if (response.data?.success && response.data?.insight) {
          setRetentionInsight(response.data.insight);
          console.log('[UserDetailsDialog] Set retention insight:', response.data.insight);
        } else {
          console.warn('[UserDetailsDialog] No insight in response:', response.data);
        }
      } catch (error: any) {
        console.error('[UserDetailsDialog] Error fetching retention insight:', error);
        console.error('[UserDetailsDialog] Error details:', {
          message: error.message,
          response: error.response?.data,
          status: error.response?.status
        });
        // Don't show error to user, just don't display insight
        setRetentionInsight(null);
      } finally {
        setLoadingInsight(false);
      }
    };

    loadResumeData();
  }, [user?.id, isOpen]);

  if (!user) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-[70vw] w-[95vw] p-0 h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader className="p-1 border-b sticky top-0 bg-white z-10 flex-shrink-0">
          <DialogTitle className="flex items-center space-x-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="h-8 w-8"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span>Candidate Profile</span>
          </DialogTitle>
        </DialogHeader>

        <ScrollArea className="flex-1 overflow-y-auto w-full">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-0 p-8">
            <div className="space-y-8">
              <div className="flex flex-col md:flex-row gap-6 items-center md:items-start">
                <div className="aspect-square w-full max-w-[250px] bg-gray-100 rounded-lg flex items-center justify-center">
                  {user.profileImage ? (
                    <img src={user.profileImage} alt="Profile" className="w-full h- object-cover rounded-lg" />
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

              <div className="space-y-5 bg-gray-50 p-1 rounded-lg">
                <div>
                  <h4 className="font-medium text-lg">Experience:</h4>
                  <p className="text-base">{user.experience}</p>
                </div>
                <div>
                  <h4 className="font-medium text-lg">Education:</h4>
                  <p className="text-base">{user.education}</p>
                </div>
                <div>
                  <h4 className="font-medium text-lg">About Me:</h4>
                  <p className="text-base text-gray-600">{user.about}</p>
                </div>
              </div>
            </div>

            <div className="space-y-1">
              <div className="bg-white rounded-lg border border-gray-200 p-2 shadow-sm">
                <h3 className="text-2xl font-semibold mb-1 text-center">AI Generated Fitment Score</h3>
                <div className="w-full max-w-[200px] mx-auto">
                  <FitmentScoreGauge score={user.score} />
                </div>

                <div className="mt-3">
                  <h4 className="font-medium text-center mb-1 text-lg">Score Breakdown</h4>
                  {(() => {
                    const overall = user.fitment_score ?? user.score ?? 0;
                    const rawType = typeof user.candidate_type === 'string' ? user.candidate_type.trim().toLowerCase() : '';
                    const isExperienced = rawType === 'experienced';
                    const datasetShare = isExperienced ? 70 : 30;
                    const big5Share = 100 - datasetShare;

                    const datasetPoints = (overall * datasetShare) / 100;
                    const big5Points = (overall * big5Share) / 100;

                    return (
                      <div className="space-y-1 px-5">
                        <div className="flex justify-between">
                          <span className="font-medium">Dataset Score:</span>
                          <span>{datasetPoints.toFixed(1)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="font-medium">Personality Score:</span>
                          <span>{big5Points.toFixed(1)}</span>
                        </div>
                      </div>
                    );
                  })()}
                </div>

                {/* Retention Analysis Subsection */}
                {retentionData && (retentionData.retention_score !== null || retentionData.retention_risk) && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <h4 className="font-medium text-center mb-2 text-lg">Likelihood to Stay Analysis</h4>
                    <div className="space-y-2 px-5">
                      {/* Retention Score
                      {retentionData.retention_score !== null && retentionData.retention_score !== undefined && (
                        <div className="flex justify-between">
                          <span className="font-medium">Retention Score:</span>
                          <span>{retentionData.retention_score.toFixed(1)}</span>
                        </div>
                      )} */}
                      
                      {/* Retention Risk Badge */}
                      {retentionData.retention_risk && (
                        <div className="flex justify-center mt-2">
                          <div className={`
                            px-4 py-1.5 rounded-full text-sm font-semibold
                            ${retentionData.retention_risk.toLowerCase() === 'low' 
                              ? 'bg-green-100 text-green-800 border border-green-300' 
                              : retentionData.retention_risk.toLowerCase() === 'medium'
                              ? 'bg-amber-100 text-amber-800 border border-amber-300'
                              : 'bg-red-100 text-red-800 border border-red-300'
                            }
                          `}>
                            <div className="flex items-center gap-1.5">
                              {retentionData.retention_risk.toLowerCase() === 'low' && (
                                <span className="text-base">✅</span>
                              )}
                              {retentionData.retention_risk.toLowerCase() === 'medium' && (
                                <span className="text-base">⚡</span>
                              )}
                              {retentionData.retention_risk.toLowerCase() === 'high' && (
                                <span className="text-base">⚠️</span>
                              )}
                              <span>Risk: {retentionData.retention_risk}</span>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* AI-Generated Retention Insight - Display right after risk badge */}
                      {loadingInsight && (
                        <div className="flex justify-center mt-2">
                          <div className="text-xs text-gray-500 italic">Generating insights...</div>
                        </div>
                      )}
                      {retentionInsight && !loadingInsight && (
                        <div className="mt-3 px-5">
                          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 shadow-sm">
                            <div className="flex items-center gap-2 mb-2">
                              <span className="text-blue-600 text-xs font-semibold uppercase tracking-wide">AI Insights</span>
                            </div>
                            <div className="text-sm text-gray-700 space-y-1.5">
                              {retentionInsight.split('\n').filter(line => line.trim().length > 0).map((line, index) => (
                                <div key={index} className="flex items-start gap-2">
                                  <span className="text-blue-600 mt-0.5">•</span>
                                  <span className="flex-1">{line.replace(/^[•\-\*]\s*/, '').trim()}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {user.personalityScores && (
                <div className="bg-white rounded-lg border border-gray-200 p-2 shadow-sm space-y-1">
                  <h4 className="text-2xl font-semibold text-center">Personality Assessment</h4>
                  <div className="w-full h-[300px] max-w-[400px] mx-auto">
                    <PersonalityPieChart scores={user.personalityScores} />
                  </div>

                  <div className="mt-6 space-y-2 max-w-md mx-auto">
                    <div className="flex justify-between text-sm">
                      <span>Extroversion</span>
                      <span>{user.personalityScores.extraversion}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Agreeableness</span>
                      <span>{user.personalityScores.agreeableness}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Openness</span>
                      <span>{user.personalityScores.openness}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Neuroticism</span>
                      <span>{user.personalityScores.neuroticism}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Conscientiousness</span>
                      <span>{user.personalityScores.conscientiousness}%</span>
                    </div>
                  </div>
                </div>
              )}

            </div>
          </div>
        </ScrollArea>

        {/* View Resume Button - Always visible at bottom */}
        <div className="p-4 border-t bg-white flex justify-center items-center flex-shrink-0">
          <Button
            onClick={() => {
              // Use resumeUrl from state (fetched from API) or fallback to user props
              const urlToOpen = resumeUrl || (user as any).resume_url || user.file_url;

              console.log('[UserDetailsDialog] View Resume button clicked:', {
                resumeUrlFromState: resumeUrl,
                resumeUrlFromUser: (user as any).resume_url,
                fileUrlFromUser: user.file_url,
                finalUrl: urlToOpen,
                userId: user.id
              });

              if (urlToOpen) {
                // If it's already a full URL (starts with http), use it directly
                const url = urlToOpen.startsWith('http')
                  ? urlToOpen
                  : urlToOpen.startsWith('/')
                    ? `${window.location.origin}${urlToOpen}`
                    : `${window.location.origin}/${urlToOpen}`;

                console.log('[UserDetailsDialog] Opening resume URL in new tab:', url);
                window.open(url, '_blank', 'noopener,noreferrer');
              } else {
                console.warn('[UserDetailsDialog] No resume URL available for user:', user.id);
                toast({
                  title: "Resume not available",
                  description: "This user doesn't have a resume uploaded.",
                  variant: "destructive"
                });
              }
            }}
            variant="default"
            size="lg"
            className="w-full max-w-md bg-blue-500 hover:bg-blue-600 text-white disabled:bg-gray-300 disabled:text-gray-500 disabled:cursor-not-allowed disabled:hover:bg-gray-300"
            disabled={!resumeUrl && !(user as any).resume_url && !user.file_url}
          >
            {resumeUrl || (user as any).resume_url || user.file_url ? "View Resume" : "No Resume Available"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}