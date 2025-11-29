
import { LineChart, Line, ResponsiveContainer } from "recharts";

interface VacancyStatsProps {
  data: Array<{ name: string; value: number }>;
}

export default function VacancyStats({ data }: VacancyStatsProps) {
  return (
    <div className="absolute bottom-0 right-0 w-full h-20 pointer-events-none">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <Line 
            type="monotone" 
            dataKey="value" 
            stroke="#fff" 
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
