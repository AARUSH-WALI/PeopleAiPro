import React, { useState, useEffect, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { CalendarIcon, PlusCircle, Clock, Loader2 } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/components/ui/use-toast";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import { format, isSameDay, parseISO, isToday } from "date-fns";
import axios from "axios";
import InterviewScheduleDialog from "@/components/dashboard/InterviewScheduleDialog";

// Simple TimePicker component
const TimePicker = ({ date, setDate }: { date: Date; setDate: (date: Date) => void }) => {
  const [hours, setHours] = useState(date.getHours());
  const [minutes, setMinutes] = useState(date.getMinutes());
  const [period, setPeriod] = useState(date.getHours() >= 12 ? 'PM' : 'AM');

  const updateTime = (h: number, m: number, p: string) => {
    let newHours = h;
    if (p === 'PM' && h < 12) newHours = h + 12;
    if (p === 'AM' && h === 12) newHours = 0;
    
    const newDate = new Date(date);
    newDate.setHours(newHours, m, 0, 0);
    setDate(newDate);
  };

  return (
    <div className="flex items-center space-x-2">
      <Select
        value={hours > 12 ? (hours - 12).toString() : (hours === 0 ? '12' : hours.toString())}
        onValueChange={(value) => {
          const newHours = parseInt(value);
          setHours(newHours);
          updateTime(newHours, minutes, period);
        }}
      >
        <SelectTrigger className="w-20">
          <SelectValue placeholder="Hour" />
        </SelectTrigger>
        <SelectContent>
          {Array.from({ length: 12 }, (_, i) => (i + 1).toString().padStart(2, '0')).map((hour) => (
            <SelectItem key={hour} value={hour}>
              {hour}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <span>:</span>
      <Select
        value={minutes.toString().padStart(2, '0')}
        onValueChange={(value) => {
          const newMinutes = parseInt(value);
          setMinutes(newMinutes);
          updateTime(hours, newMinutes, period);
        }}
      >
        <SelectTrigger className="w-20">
          <SelectValue placeholder="Min" />
        </SelectTrigger>
        <SelectContent>
          {['00', '15', '30', '45'].map((minute) => (
            <SelectItem key={minute} value={minute}>
              {minute}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Select
        value={period}
        onValueChange={(value) => {
          setPeriod(value);
          updateTime(hours, minutes, value);
        }}
      >
        <SelectTrigger className="w-20">
          <SelectValue placeholder="AM/PM" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="AM">AM</SelectItem>
          <SelectItem value="PM">PM</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
};

interface Interview {
  id: string;
  title: string;
  description: string;
  candidateId: string;
  candidateName: string;
  candidateEmail: string;
  interviewerId: string;
  interviewerName: string;
  interviewerEmail: string;
  date: string;
  status: 'scheduled' | 'completed' | 'cancelled';
  meetingLink?: string;
  meetingPlatform: string;
  interviewType: string;
  jobTitle?: string;
  jobDescription?: string;
  interviewerNotes?: string;
  candidateFeedback?: string;
  technicalAssessment?: string;
  overallRating?: number;
  createdAt: string;
  updatedAt: string;
}

interface Candidate {
  id: string;
  name: string;
  email: string;
  jobRole?: string;
}

// Use production API URL in production environment, otherwise use development URL
const API_BASE_URL = import.meta.env.PROD 
  ? import.meta.env.VITE_PROD_API_BASE_URL 
  : import.meta.env.VITE_DEV_API_BASE_URL || 'http://localhost:5000/api';

export default function Interview() {
  const { toast } = useToast();
  const [interviews, setInterviews] = useState<Interview[]>([]);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [showEventDialog, setShowEventDialog] = useState(false);
  const [showScheduleDialog, setShowScheduleDialog] = useState(false);
  const [showInterviewDetails, setShowInterviewDetails] = useState(false);
  const [selectedInterview, setSelectedInterview] = useState<Interview | null>(null);
  const [selectedDate, setSelectedDate] = useState<Date>(new Date());
  const [calendarMonth, setCalendarMonth] = useState<Date>(new Date());
  const [newEvent, setNewEvent] = useState<{
    title: string;
    description: string;
    candidateId: string;
    date: Date;
    interviewType: string;
    meetingPlatform: string;
    meetingLink: string;
    interviewerId: string;
    interviewerName: string;
    interviewerEmail: string;
  }>({
    title: '',
    description: '',
    candidateId: '',
    date: new Date(),
    interviewType: 'technical',
    meetingPlatform: 'Zoom',
    meetingLink: '',
    interviewerId: '',
    interviewerName: '',
    interviewerEmail: ''
  });

  // Fetch interviews and candidates
  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        console.log('Fetching data from:', `${API_BASE_URL}/resumes`);
        
        // Fetch candidates first
        const candidatesRes = await axios.get(`${API_BASE_URL}/resumes`);
        console.log('Candidates API Response:', candidatesRes.data);

        if (candidatesRes.data?.data) {
          const candidatesData = Array.isArray(candidatesRes.data.data) 
            ? candidatesRes.data.data 
            : [candidatesRes.data.data];
            
          console.log('Processed candidates data:', candidatesData);
          
          const formattedCandidates = candidatesData.map((c: any) => ({
            id: c.id || c.email, // Use email as fallback ID if id is not present
            name: c.name || 'Unknown Candidate',
            email: c.email || '',
            jobRole: c.best_fit_for || c.job_title || 'Not specified'
          }));
          
          console.log('Formatted candidates:', formattedCandidates);
          setCandidates(formattedCandidates);
        }

        // Then fetch interviews
        try {
          const interviewsRes = await axios.get(`${API_BASE_URL}/interviews`);
          console.log('Interviews API Response:', interviewsRes);
          
          if (interviewsRes.data?.data) {
            const interviewsData = Array.isArray(interviewsRes.data.data) 
              ? interviewsRes.data.data 
              : [interviewsRes.data.data];
            
            // Transform the interview data to match our interface
            const formattedInterviews = interviewsData.map((interview: any) => ({
              id: interview.id || '',
              title: interview.title || 'Untitled Interview',
              description: interview.description || '',
              candidateId: interview.candidate_id || interview.candidateId || '',
              candidateName: interview.candidate_name || interview.candidateName || 'Unknown Candidate',
              candidateEmail: interview.candidate_email || interview.candidateEmail || '',
              interviewerId: interview.interviewer_id || interview.interviewerId || '',
              interviewerName: interview.interviewer_name || interview.interviewerName || '',
              interviewerEmail: interview.interviewer_email || interview.interviewerEmail || '',
              date: interview.date || new Date().toISOString(),
              status: interview.status || 'scheduled',
              meetingLink: interview.meeting_link || interview.meetingLink || '',
              meetingPlatform: interview.meeting_platform || interview.meetingPlatform || '',
              interviewType: interview.interview_type || interview.interviewType || '',
              jobTitle: interview.job_title || interview.jobTitle || '',
              jobDescription: interview.job_description || interview.jobDescription || '',
              interviewerNotes: interview.interviewer_notes || interview.interviewerNotes || '',
              candidateFeedback: interview.candidate_feedback || interview.candidateFeedback || '',
              technicalAssessment: interview.technical_assessment || interview.technicalAssessment || '',
              overallRating: interview.overall_rating || interview.overallRating || 0,
              createdAt: interview.created_at || interview.createdAt || new Date().toISOString(),
              updatedAt: interview.updated_at || interview.updatedAt || new Date().toISOString()
            }));
            
            console.log('Formatted interviews:', formattedInterviews);
            setInterviews(formattedInterviews);
          }
        } catch (interviewError) {
          console.error('Error fetching interviews:', interviewError);
          toast({
            title: "Warning",
            description: "Interviews could not be loaded, but you can still schedule new ones.",
            variant: "default"
          });
        }
      } catch (error) {
        console.error('Error in fetchData:', error);
        const errorMessage = error.response?.data?.message || error.message || 'Unknown error';
        console.error('Error details:', {
          status: error.response?.status,
          data: error.response?.data,
          config: {
            url: error.config?.url,
            method: error.config?.method,
          },
        });
        
        toast({
          title: "Error",
          description: `Failed to fetch data: ${errorMessage}`,
          variant: "destructive"
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  const handleCreateEvent = async () => {
    // Get the selected candidate
    const selectedCandidate = candidates.find(c => c.id === newEvent.candidateId);
    
    if (!selectedCandidate || !newEvent.candidateId) {
      toast({
        title: "Error",
        description: "Please select a valid candidate",
        variant: "destructive"
      });
      return;
    }

    // Validate all required fields
    if (!newEvent.title || !newEvent.description) {
      toast({
        title: "Missing Required Fields",
        description: "Please fill in all required fields to schedule an interview",
        variant: "destructive"
      });
      return;
    }

    try {
      setIsCreating(true);
      const interviewData = {
        title: newEvent.title,
        description: newEvent.description,
        candidateId: newEvent.candidateId,
        candidateName: selectedCandidate.name,
        candidateEmail: selectedCandidate.email,
        date: newEvent.date.toISOString(),
        status: 'scheduled',
        interviewerId: newEvent.interviewerId || null,
        interviewerName: newEvent.interviewerName || null,
        interviewerEmail: newEvent.interviewerEmail || null,
        meetingLink: newEvent.meetingLink || null,
        meetingPlatform: newEvent.meetingPlatform,
        interviewType: newEvent.interviewType,
        jobTitle: '',
        jobDescription: '',
        interviewerNotes: '',
        candidateFeedback: '',
        technicalAssessment: '',
        overallRating: 0
      };

      console.log('Submitting interview data:', interviewData);
      
      const response = await axios.post(`${API_BASE_URL}/interviews`, interviewData);

      if (response.data?.success) {
        setInterviews([...interviews, response.data.data]);
        toast({
          title: "Success",
          description: `Interview scheduled successfully for ${format(newEvent.date, 'PPP')}`
        });
        setShowEventDialog(false);
        setNewEvent({
          title: '',
          description: '',
          candidateId: '',
          date: new Date(),
          interviewType: 'technical',
          meetingPlatform: 'Zoom',
          meetingLink: '',
          interviewerId: '',
          interviewerName: '',
          interviewerEmail: ''
        });
      }
    } catch (error) {
      console.error('Error creating interview:', error);
      toast({
        title: "Error",
        description: "Failed to schedule interview. Please try again.",
        variant: "destructive"
      });
    } finally {
      setIsCreating(false);
    }
  };

  // Function to get interviews for a specific date
  const getInterviewsForDate = (date: Date) => {
    if (!interviews || interviews.length === 0) return [];
    
    return interviews.filter(interview => {
      try {
        if (!interview || !interview.date) return false;
        const interviewDate = new Date(interview.date);
        return isSameDay(interviewDate, date);
      } catch (error) {
        console.error('Error processing interview date:', error, interview);
        return false;
      }
    });
  };
  
  // Get unique interview dates for calendar highlighting
  const interviewDates = useMemo<Date[]>(() => {
    if (!interviews || interviews.length === 0) return [];
    
    const dates = interviews
      .map(interview => {
        try {
          return interview?.date ? new Date(interview.date) : null;
        } catch (error) {
          console.error('Error parsing interview date:', error);
          return null;
        }
      })
      .filter((date): date is Date => date !== null);
    
    // Remove duplicates and return
    return Array.from(new Set(dates.map(date => date.toDateString())))
      .map(dateStr => new Date(dateStr));
  }, [interviews]);

  // Get interviews for the selected date and future dates
  const upcomingInterviews = interviews
    .filter(interview => {
      try {
        return interview && interview.date && new Date(interview.date) >= new Date();
      } catch (error) {
        console.error('Error filtering upcoming interviews:', error, interview);
        return false;
      }
    })
    .sort((a, b) => {
      try {
        return new Date(a.date).getTime() - new Date(b.date).getTime();
      } catch (error) {
        console.error('Error sorting interviews:', error);
        return 0;
      }
    });

  // Get interviews for today and selected date
  const todayInterviews = getInterviewsForDate(new Date());
  const selectedDateInterviews = getInterviewsForDate(selectedDate);

  // Handle interview click to show details
  const handleInterviewClick = (interview: Interview) => {
    setSelectedInterview(interview);
    setShowInterviewDetails(true);
  };

  return (
    <div className="page-container bg-gradient-to-br from-purple-50/80 to-white p-6">
      <h1 className="text-2xl font-bold text-purple-800 mb-6">Interview Calendar</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Calendar View */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
            <h3 className="font-medium text-lg mb-4 text-gray-800">Calendar View</h3>
            <Calendar
              mode="single"
              selected={selectedDate}
              onSelect={(date) => date && setSelectedDate(date)}
              month={calendarMonth}
              onMonthChange={setCalendarMonth}
              className="w-full"
              disabled={{ before: new Date() }}
              modifiers={{
                hasInterviews: interviewDates
              }}
              modifiersStyles={{
                hasInterviews: {
                  border: '2px solid #8b5cf6',
                  borderRadius: '50%'
                }
              }}
            />
          </div>

          {/* Upcoming Interviews */}
          <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
            <h3 className="font-medium text-lg mb-4 text-gray-800">Upcoming Interviews</h3>
            {upcomingInterviews.length > 0 ? (
              <div className="space-y-3">
                {upcomingInterviews.map((interview) => (
                  <div 
                    key={interview.id}
                    onClick={() => handleInterviewClick(interview)}
                    className="p-4 bg-white rounded-lg border border-gray-200 hover:border-purple-300 cursor-pointer transition-colors"
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium text-gray-900">{interview.title}</h4>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            interview.status === 'scheduled' 
                              ? 'bg-blue-100 text-blue-800'  
                              : interview.status === 'completed'
                              ? 'bg-green-100 text-green-800'
                              : 'bg-red-100 text-red-800'
                          }`}>
                            {interview.status.charAt(0).toUpperCase() + interview.status.slice(1)}
                          </span>
                        </div>
                        
                        <div className="mt-2 space-y-1">
                          <div className="flex items-start">
                            <Clock className="h-4 w-4 text-gray-400 mt-0.5 mr-2 flex-shrink-0" />
                            <p className="text-sm text-gray-600">
                              {format(new Date(interview.date), 'EEEE, MMMM d, yyyy • h:mm a')}
                            </p>
                          </div>
                          
                          <div className="flex items-start">
                            <svg className="h-4 w-4 text-gray-400 mt-0.5 mr-2 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                            </svg>
                            <p className="text-sm text-gray-600">
                              <span className="font-medium">Candidate:</span> {interview.candidateName}
                              {interview.candidateEmail && ` (${interview.candidateEmail})`}
                            </p>
                          </div>
                          
                          {interview.interviewerName && (
                            <div className="flex items-start">
                              <svg className="h-4 w-4 text-gray-400 mt-0.5 mr-2 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                              </svg>
                              <p className="text-sm text-gray-600">
                                <span className="font-medium">Interviewer:</span> {interview.interviewerName}
                                {interview.interviewerEmail && ` (${interview.interviewerEmail})`}
                              </p>
                            </div>
                          )}
                          
                          <div className="flex items-start">
                            <svg className="h-4 w-4 text-gray-400 mt-0.5 mr-2 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                            </svg>
                            <div className="flex flex-wrap gap-2">
                              {interview.interviewType && (
                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                                  {interview.interviewType}
                                </span>
                              )}
                              {interview.meetingPlatform && (
                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                                  {interview.meetingPlatform}
                                </span>
                              )}
                              {interview.meetingLink && (
                                <a 
                                  href={interview.meetingLink} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800 hover:bg-green-200"
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  Join Meeting
                                </a>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-4 bg-gray-50 rounded-lg border border-gray-200 text-center">
                <p className="text-sm text-gray-500">
                  No upcoming interviews scheduled.
                </p>
              </div>
            )}
          </div>
        </div>
        
        {/* Selected Date's Interviews */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg border border-gray-200 p-4 shadow-sm sticky top-4">
            <h3 className="font-medium text-lg mb-4 text-gray-800">
              {format(selectedDate, 'EEEE, MMMM d, yyyy')}
            </h3>
            
            {isLoading ? (
              <div className="flex justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-purple-600" />
              </div>
            ) : selectedDateInterviews.length > 0 ? (
              <div className="space-y-3">
                {selectedDateInterviews.map((interview) => (
                  <div 
                    key={interview.id}
                    onClick={() => handleInterviewClick(interview)}
                    className="p-3 bg-purple-50 rounded-lg border border-purple-100 hover:border-purple-300 cursor-pointer transition-colors"
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <h4 className="font-medium text-purple-900">{interview.title}</h4>
                        <p className="text-sm text-gray-600 mt-1">
                          {format(new Date(interview.date), 'h:mm a')} • {interview.candidateName}
                        </p>
                      </div>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        interview.status === 'scheduled' 
                          ? 'bg-blue-100 text-blue-800'  
                          : interview.status === 'completed'
                          ? 'bg-green-100 text-green-800'
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {interview.status.charAt(0).toUpperCase() + interview.status.slice(1)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-4 bg-purple-50 rounded-lg border border-purple-100 text-center">
                <p className="text-sm text-gray-500">
                  No interviews scheduled for {isToday(selectedDate) ? 'today' : 'this date'}.
                </p>
              </div>
            )}
            
            <Button 
              onClick={() => {
                setNewEvent(prev => ({ ...prev, date: selectedDate }));
                setShowEventDialog(true);
              }}
              className="w-full bg-gradient-to-r from-purple-500 to-purple-700 hover:from-purple-600 hover:to-purple-800 mt-4"
            >
              <CalendarIcon className="h-4 w-4 mr-2" />
              Schedule Interview
            </Button>
          </div>
        </div>
      </div>

      {/* Interview Details Dialog */}
      <Dialog open={showInterviewDetails} onOpenChange={setShowInterviewDetails}>
        <DialogContent className="sm:max-w-[500px]">
          {selectedInterview && (
            <>
              <DialogHeader>
                <DialogTitle className="text-xl">{selectedInterview.title}</DialogTitle>
                <div className="flex items-center gap-2 text-sm text-gray-500 mt-1">
                  <CalendarIcon className="h-4 w-4" />
                  <span>{format(new Date(selectedInterview.date), 'EEEE, MMMM d, yyyy • h:mm a')}</span>
                </div>
              </DialogHeader>
              
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <h4 className="font-medium">Candidate</h4>
                  <div className="p-3 bg-gray-50 rounded-md">
                    <p className="font-medium">{selectedInterview.candidateName}</p>
                    <p className="text-sm text-gray-600">{selectedInterview.candidateEmail}</p>
                  </div>
                </div>

                {selectedInterview.interviewerName && (
                  <div className="space-y-2">
                    <h4 className="font-medium">Interviewer</h4>
                    <div className="p-3 bg-gray-50 rounded-md">
                      <p className="font-medium">{selectedInterview.interviewerName}</p>
                      <p className="text-sm text-gray-600">{selectedInterview.interviewerEmail}</p>
                    </div>
                  </div>
                )}

                <div className="space-y-2">
                  <h4 className="font-medium">Details</h4>
                  <div className="p-3 bg-gray-50 rounded-md space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-500">Type:</span>
                      <span className="font-medium">
                        {selectedInterview.interviewType || 'Not specified'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-500">Platform:</span>
                      <span className="font-medium">
                        {selectedInterview.meetingPlatform || 'Not specified'}
                      </span>
                    </div>
                    {selectedInterview.meetingLink && (
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-500">Meeting Link:</span>
                        <a 
                          href={selectedInterview.meetingLink} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-purple-600 hover:underline text-sm"
                        >
                          Join Meeting
                        </a>
                      </div>
                    )}
                  </div>
                </div>

                {selectedInterview.description && (
                  <div className="space-y-2">
                    <h4 className="font-medium">Description</h4>
                    <div className="p-3 bg-gray-50 rounded-md">
                      <p className="text-sm text-gray-700 whitespace-pre-line">
                        {selectedInterview.description}
                      </p>
                    </div>
                  </div>
                )}

                <div className="flex justify-end gap-2 pt-4">
                  <Button 
                    variant="outline" 
                    onClick={() => setShowInterviewDetails(false)}
                  >
                    Close
                  </Button>
                  <Button 
                    variant="default"
                    onClick={() => {
                      // Add edit functionality here
                      setShowInterviewDetails(false);
                    }}
                  >
                    Edit Interview
                  </Button>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* Event creation dialog */}
      <Dialog open={showEventDialog} onOpenChange={setShowEventDialog}>
        <DialogContent className="bg-white border border-purple-100 sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="text-purple-800">Schedule Interview for {format(selectedDate, 'PPP')}</DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium text-gray-700 mb-1 block">Interview Title *</label>
                <Input
                  placeholder="E.g., Technical Interview - Frontend"
                  value={newEvent.title}
                  onChange={(e) => setNewEvent({ ...newEvent, title: e.target.value })}
                  className="border-purple-200 focus-visible:ring-purple-500"
                  required
                />
              </div>
              
              <div>
                <label className="text-sm font-medium text-gray-700 mb-1 block">Interview Description *</label>
                <Textarea
                  placeholder="Enter interview details, topics to cover, etc."
                  value={newEvent.description}
                  onChange={(e) => setNewEvent({ ...newEvent, description: e.target.value })}
                  className="min-h-[100px] border-purple-200 focus-visible:ring-purple-500"
                  required
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-1 block">Date & Time *</label>
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        className={`w-full justify-start text-left font-normal ${!newEvent.date ? 'text-muted-foreground' : ''}`}
                      >
                        <CalendarIcon className="mr-2 h-4 w-4" />
                        {newEvent.date ? format(newEvent.date, "PPPp") : <span>Pick a date and time</span>}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0">
                      <Calendar
                        mode="single"
                        selected={newEvent.date}
                        onSelect={(date) => date && setNewEvent({...newEvent, date})}
                        initialFocus
                      />
                      <div className="p-3 border-t">
                        <TimePicker 
                          date={newEvent.date} 
                          setDate={(date) => setNewEvent({...newEvent, date})} 
                        />
                      </div>
                    </PopoverContent>
                  </Popover>
                </div>

                <div>
                  <label className="text-sm font-medium text-gray-700 mb-1 block">Interview Type *</label>
                  <Select
                    value={newEvent.interviewType || 'technical'}
                    onValueChange={(value) => setNewEvent({...newEvent, interviewType: value})}
                  >
                    <SelectTrigger className="w-full border-purple-200 focus:ring-purple-500">
                      <SelectValue placeholder="Select interview type" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="technical">Technical</SelectItem>
                      <SelectItem value="hr">HR</SelectItem>
                      <SelectItem value="manager">Manager</SelectItem>
                      <SelectItem value="culture">Culture Fit</SelectItem>
                      <SelectItem value="other">Other</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>


              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-1 block">Meeting Platform *</label>
                  <Select
                    value={newEvent.meetingPlatform || 'Zoom'}
                    onValueChange={(value) => setNewEvent({...newEvent, meetingPlatform: value})}
                  >
                    <SelectTrigger className="w-full border-purple-200 focus:ring-purple-500">
                      <SelectValue placeholder="Select platform" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Zoom">Zoom</SelectItem>
                      <SelectItem value="Google Meet">Google Meet</SelectItem>
                      <SelectItem value="Microsoft Teams">Microsoft Teams</SelectItem>
                      <SelectItem value="In-Person">In-Person</SelectItem>
                      <SelectItem value="Other">Other</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <label className="text-sm font-medium text-gray-700 mb-1 block">Meeting Link (if online)</label>
                  <Input
                    placeholder="https://meet.example.com/your-meeting"
                    value={newEvent.meetingLink || ''}
                    onChange={(e) => setNewEvent({ ...newEvent, meetingLink: e.target.value })}
                    className="border-purple-200 focus-visible:ring-purple-500"
                  />
                </div>
              </div>
            </div>
            
            <div>
              <div className="space-y-2">
                <Select
                  value={newEvent.candidateId}
                  onValueChange={(value) => setNewEvent({ ...newEvent, candidateId: value })}
                  disabled={isLoading || candidates.length === 0}
                >
                  <SelectTrigger className="border-purple-200 focus:ring-purple-500">
                    <SelectValue 
                      placeholder={
                        isLoading 
                          ? 'Loading candidates...' 
                          : candidates.length === 0
                          ? 'No candidates available'
                          : 'Select Candidate'
                      } 
                    />
                  </SelectTrigger>
                  <SelectContent className="bg-white border border-purple-100 max-h-60 overflow-y-auto">
                    {candidates.length > 0 ? (
                      candidates.map((candidate) => (
                        <SelectItem key={candidate.id} value={candidate.id}>
                          <div className="flex flex-col">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">{candidate.name}</span>
                              {candidate.email && (
                                <span className="text-xs text-gray-500 truncate max-w-[120px]" title={candidate.email}>
                                  ({candidate.email})
                                </span>
                              )}
                            </div>
                            {candidate.jobRole && (
                              <span className="text-xs text-gray-500">
                                {candidate.jobRole}
                              </span>
                            )}
                          </div>
                        </SelectItem>
                      ))
                    ) : (
                      <div className="py-2 px-3 text-sm text-gray-500">
                        No candidates found. Please add candidates first.
                      </div>
                    )}
                  </SelectContent>
                </Select>
                {isLoading ? (
                  <p className="text-xs text-gray-500 flex items-center gap-1">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    Loading candidates...
                  </p>
                ) : candidates.length === 0 ? (
                  <p className="text-xs text-amber-600">
                    No candidates available. Please add candidates first.
                  </p>
                ) : (
                  <p className="text-xs text-gray-500">
                    {candidates.length} candidate{candidates.length !== 1 ? 's' : ''} available
                  </p>
                )}
              </div>
            </div>
            
            <div className="flex justify-end space-x-2 pt-4">
              <Button 
                variant="outline" 
                onClick={() => setShowEventDialog(false)}
                className="border-purple-200 text-purple-700 hover:bg-purple-50"
              >
                Cancel
              </Button>
              <Button 
                onClick={handleCreateEvent}
                className="bg-purple-600 hover:bg-purple-700"
                disabled={isCreating}
              >
                {isCreating ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Scheduling...
                  </>
                ) : (
                  'Schedule Interview'
                )}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
      
      {/* Today's schedule dialog */}
      <InterviewScheduleDialog 
        isOpen={showScheduleDialog} 
        onClose={() => setShowScheduleDialog(false)}
        interviews={todayInterviews}
        isLoading={isLoading}
      />
    </div>
  );
}
