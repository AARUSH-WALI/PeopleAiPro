
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { format, parseISO } from "date-fns";
import { Clock, Loader2 } from "lucide-react";

interface Interview {
  id: string;
  title: string;
  candidateName: string;
  candidateEmail: string;
  date: string;
  status: 'scheduled' | 'completed' | 'cancelled';
  description?: string;
}

interface InterviewScheduleDialogProps {
  isOpen: boolean;
  onClose: () => void;
  interviews: Interview[];
  isLoading: boolean;
}

export default function InterviewScheduleDialog({ 
  isOpen, 
  onClose, 
  interviews, 
  isLoading 
}: InterviewScheduleDialogProps) {

  // Ensure interviews is always an array
  const safeInterviews = Array.isArray(interviews) ? interviews : [];

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[425px] bg-white border border-gray-200 text-gray-800">
        <DialogHeader className="bg-purple-50 -mx-6 -mt-6 px-6 py-4 rounded-t-xl">
          <DialogTitle className="text-purple-700 flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Today's Interview Schedule
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-3 mt-2">
          {isLoading ? (
            <div className="flex justify-center items-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-purple-600" />
            </div>
          ) : safeInterviews.length > 0 ? (
            safeInterviews.map((interview) => (
              <div 
                key={interview.id}
                className="flex justify-between items-start p-3 bg-gradient-to-r from-gray-50 to-white rounded-lg border border-gray-200 hover:shadow-sm hover:border-purple-200 transition-all"
              >
                <div>
                  <p className="font-medium text-gray-800">{interview.candidateName}</p>
                  <p className="text-sm text-gray-500">{interview.title}</p>
                  {interview.description && (
                    <p className="text-xs text-gray-500 mt-1 line-clamp-1">
                      {interview.description}
                    </p>
                  )}
                </div>
                <div className="flex flex-col items-end gap-1">
                  <div className="bg-purple-50 px-3 py-1 rounded-full text-sm font-medium text-purple-700 border border-purple-100">
                    {format(parseISO(interview.date), "h:mm a")}
                  </div>
                  <span className={`px-2 py-0.5 text-xs rounded-full ${
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
            ))
          ) : (
            <div className="text-center py-6">
              <p className="text-gray-500">No interviews scheduled for today.</p>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
