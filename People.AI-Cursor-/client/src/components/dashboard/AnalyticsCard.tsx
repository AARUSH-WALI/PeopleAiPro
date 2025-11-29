
import { Area, AreaChart, ResponsiveContainer } from "recharts";
import { MoreHorizontal } from "lucide-react";

interface Candidate {
  name: string;
  title: string;
  score: number;
  trend: number[];
}

interface AnalyticsCardProps {
  candidates: Candidate[];
}

export default function AnalyticsCard({ candidates }: AnalyticsCardProps) {
  return (
    <div className="bg-white rounded-lg p-4 shadow-sm">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">Big 5 Analysis</h3>
        <button className="p-1 rounded-md hover:bg-gray-100">
          <MoreHorizontal size={20} />
        </button>
      </div>
      
      <div className="space-y-4">
        {candidates.map((candidate) => (
          <div key={candidate.name} className="flex items-center space-x-4">
            <div className="flex-1">
              <div className="font-medium">{candidate.name}</div>
              <div className="text-sm text-gray-500">{candidate.title}</div>
            </div>
            
            <div className="w-24 h-12">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={candidate.trend.map((value, i) => ({ value, i }))}>
                  <Area 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#9b87f5" 
                    fill="#9b87f5" 
                    fillOpacity={0.2} 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            
            <div className="text-right font-semibold w-16">{candidate.score.toFixed(2)}</div>
            
            <div className="w-5 h-5 bg-green-500 rounded-full flex items-center justify-center">
              <span className="text-white text-xs">✓</span>
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-4 text-right">
        <a href="#" className="text-blue-500 text-sm font-medium">View All →</a>
      </div>
    </div>
  );
}
