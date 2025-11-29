
// import { Button } from "@/components/ui/button";
// import { ArrowRight } from "lucide-react";
// import { useState } from "react";
// import { Link } from "react-router-dom";
// import CandidateDetailsDialog from "./CandidateDetailsDialog";

// export interface Candidate {
//   id: string;
//   name: string;
//   email: string;
//   fitment_score: number;
//   phone?: string;
//   education?: string;
//   experience?: string;
//   skills?: string[];
//   projects?: string[];
//   expectedRole?: string;
//   location?: string;
//   status?: string;
// }

// interface CandidateScoresProps {
//   candidates: Candidate[];
//   onViewUser?: (candidate: Candidate) => void;
// }

// export default function CandidateScores({ candidates, onViewUser }: CandidateScoresProps) {
//   const [selectedCandidate, setSelectedCandidate] = useState<Candidate | null>(null);

//   const handleCandidateClick = (candidate: Candidate) => {
//     if (onViewUser) {
//       onViewUser(candidate);
//     } else {
//       setSelectedCandidate(candidate);
//     }
//   };

//   return (
//     <div className="bg-white p-6 rounded-lg shadow-lg h-full flex flex-col">
//       <div className="flex justify-between items-center mb-4">
//         <h3 className="text-2xl font-extrabold text-indigo-600">
//           Candidate Fitment Scores
//         </h3>
//         <Button variant="ghost" size="sm" asChild>
//           <Link to="/users" className="flex items-center gap-2">
//             View All
//             <ArrowRight size={16} />
//           </Link>
//         </Button>
//       </div>

//       <div className="space-y-4 flex-grow">
//         {candidates.map((candidate) => (
//           <div
//             key={candidate.email}
//             className="p-4 rounded-lg border border-gray-200 hover:border-gray-300 transition-all cursor-pointer"
//             onClick={() => handleCandidateClick(candidate)}
//           >
//             <div className="flex justify-between items-center">
//               <div>
//                 <h4 className="font-medium text-gray-900">{candidate.name}</h4>
//                 <p className="text-sm text-gray-500">{candidate.email}</p>
//               </div>
//               <div
//                 className={`text-lg font-semibold ${
//                   candidate.fitment_score >= 60
//                     ? "text-green-600"
//                     : "text-yellow-600"
//                 }`}
//               >
//                 {candidate.fitment_score.toFixed(1)}%
//               </div>
//             </div>

//             {/* Green Bar under candidates with score greater than 60% */}
//             {candidate.fitment_score >= 60 && (
//               <div className="mt-2 w-full bg-green-100 rounded-full h-2.5">
//                 <div
//                   className="h-2.5 rounded-full bg-green-500"
//                   style={{ width: `${candidate.fitment_score}%` }}
//                 ></div>
//               </div>
//             )}

//             {/* Yellow Bar under candidates with score less than 60% */}
//             {candidate.fitment_score < 60 && (
//               <div className="mt-2 w-full bg-yellow-100 rounded-full h-2.5">
//                 <div
//                   className="h-2.5 rounded-full bg-yellow-500"
//                   style={{ width: `${candidate.fitment_score}%` }}
//                 ></div>
//               </div>
//             )}
//           </div>
//         ))}
//       </div>

//       <CandidateDetailsDialog
//         isOpen={!!selectedCandidate}
//         onClose={() => setSelectedCandidate(null)}
//         candidate={selectedCandidate}
//       />
//     </div>
//   );
// }

import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";

export interface Candidate {
  id: string;
  name: string;
  email: string;
  fitment_score: number;
  phone?: string;
  education?: string;
  experience?: string;
  skills?: string[];
  projects?: string[];
  expectedRole?: string;
  location?: string;
  status?: string;
}

interface CandidateScoresProps {
  candidates: Candidate[];
  onViewUser?: (candidate: Candidate) => void;
}

export default function CandidateScores({ candidates, onViewUser }: CandidateScoresProps) {
  const navigate = useNavigate();

  const handleCandidateClick = (candidate: Candidate) => {
    if (candidate.id) {
      navigate(`/users?selected=${candidate.id}`);
    } else if (onViewUser) {
      onViewUser(candidate);
    }
  };

  // ✅ Sort by score (descending) and take top 5
  const topCandidates = [...candidates]
    .sort((a, b) => b.fitment_score - a.fitment_score)
    .slice(0, 5);

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-2xl font-extrabold text-indigo-600">
          Top 5 Candidate Fitment Scores
        </h3>
        <Button variant="ghost" size="sm" asChild>
          <Link to="/users" className="flex items-center gap-2">
            View All
            <ArrowRight size={16} />
          </Link>
        </Button>
      </div>

      <div className="space-y-4 flex-grow">
        {topCandidates.map((candidate) => (
          <div
            key={candidate.email}
            className="p-4 rounded-lg border border-gray-200 hover:border-gray-300 transition-all cursor-pointer"
            onClick={() => handleCandidateClick(candidate)}
          >
            <div className="flex justify-between items-center">
              <div>
                <h4 className="font-medium text-gray-900">{candidate.name}</h4>
                <p className="text-sm text-gray-500">{candidate.email}</p>
              </div>
              <div
                className={`text-lg font-semibold ${candidate.fitment_score >= 65
                  ? "text-green-600"
                  : "text-yellow-600"
                  }`}
              >
                {candidate.fitment_score.toFixed(1)}%
              </div>
            </div>

            {/* Progress Bar */}
            <div className="mt-2 w-full rounded-full h-2.5 bg-gray-100">
              <div
                className={`h-2.5 rounded-full ${candidate.fitment_score >= 65
                  ? "bg-green-500"
                  : "bg-yellow-500"
                  }`}
                style={{ width: `${candidate.fitment_score}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
