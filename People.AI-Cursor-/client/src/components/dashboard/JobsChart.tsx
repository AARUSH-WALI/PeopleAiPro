
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const data = [
  { name: "Jan", onHold: 30, likelyFit: 25, fit: 50, rejected: 0 },
  { name: "Feb", onHold: 130, likelyFit: 50, fit: 100, rejected: 0 },
  { name: "Mar", onHold: 40, likelyFit: 20, fit: 20, rejected: 80 },
  { name: "Apr", onHold: 40, likelyFit: 20, fit: 45, rejected: 0 },
  { name: "May", onHold: 45, likelyFit: 55, fit: 15, rejected: 0 },
  { name: "Jun", onHold: 80, likelyFit: 40, fit: 100, rejected: 0 },
  { name: "Jul", onHold: 45, likelyFit: 75, fit: 90, rejected: 0 },
  { name: "Aug", onHold: 30, likelyFit: 20, fit: 0, rejected: 0 },
  { name: "Sep", onHold: 40, likelyFit: 25, fit: 50, rejected: 0 },
  { name: "Oct", onHold: 55, likelyFit: 80, fit: 35, rejected: 0 },
  { name: "Nov", onHold: 30, likelyFit: 10, fit: 30, rejected: 150 },
  { name: "Dec", onHold: 75, likelyFit: 60, fit: 20, rejected: 0 },
];

export default function JobsChart() {
  const [filter, setFilter] = useState<"Today" | "Month" | "Year">("Today");
  
  return (
    <div className="bg-white rounded-lg p-4 shadow-sm">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">Hiring Ratio</h3>
        <div className="flex space-x-2">
          <FilterButton 
            active={filter === "Today"} 
            onClick={() => setFilter("Today")}
          >
            Today
          </FilterButton>
        </div>
      </div>
      
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="onHold" stackId="a" fill="#B8E0FF" name="On Hold" />
            <Bar dataKey="likelyFit" stackId="a" fill="#4299E1" name="Likely Fit" />
            <Bar dataKey="fit" stackId="a" fill="#7E69AB" name="Fit" />
            <Bar dataKey="rejected" stackId="a" fill="#E2E8F0" name="Rejected" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function FilterButton({ 
  children, 
  active, 
  onClick 
}: { 
  children: React.ReactNode, 
  active: boolean, 
  onClick: () => void 
}) {
  return (
    <Button
      variant="outline"
      size="sm"
      onClick={onClick}
      className={cn(
        "border-gray-200 text-sm",
        active ? "bg-gray-100" : "bg-white"
      )}
    >
      {children}
    </Button>
  );
}
