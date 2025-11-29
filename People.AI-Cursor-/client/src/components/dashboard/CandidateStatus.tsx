
import React from "react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts";

interface CandidateStatusProps {
  approvedCount: number;
  reviewCount: number;
  rejectedCount: number;
}

export default function CandidateStatus({
  approvedCount = 45,
  reviewCount = 30,
  rejectedCount = 25,
}: CandidateStatusProps) {
  const total = approvedCount + reviewCount + rejectedCount;
  
  const data = [
    { 
      name: "Approved", 
      value: approvedCount, 
      percentage: Math.round((approvedCount / total) * 100),
      color: "#9b87f5" 
    },
    { 
      name: "Under Review", 
      value: reviewCount, 
      percentage: Math.round((reviewCount / total) * 100),
      color: "#4da3ff" 
    },
    { 
      name: "Rejected", 
      value: rejectedCount, 
      percentage: Math.round((rejectedCount / total) * 100),
      color: "#ff6b6b" 
    },
  ];
  
  const RADIAN = Math.PI / 180;
  const renderCustomizedLabel = ({
    cx,
    cy,
    midAngle,
    innerRadius,
    outerRadius,
    percent,
    index,
  }: any) => {
    const radius = innerRadius + (outerRadius - innerRadius) * 0.6;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
      <text
        x={x}
        y={y}
        fill="#ffffff"
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={14}
        fontWeight="bold"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-200">
      <div className="p-5 border-b border-gray-200 bg-gradient-to-r from-purple-50 to-white">
        <h2 className="text-xl font-bold text-gray-800">Candidate Status</h2>
      </div>
      <div className="p-5">
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={renderCustomizedLabel}
                outerRadius={80}
                innerRadius={40}
                dataKey="value"
                strokeWidth={5}
                stroke="#ffffff"
              >
                {data.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.color}
                  />
                ))}
              </Pie>
              <Tooltip 
                formatter={(value: number, name: string, props: any) => {
                  return [`${value} (${props.payload.percentage}%)`, name];
                }}
                contentStyle={{ borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}
              />
              <Legend verticalAlign="bottom" align="center" />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="flex flex-wrap justify-center gap-6 mt-4">
          {data.map((item) => (
            <div key={item.name} className="text-center">
              <div className="text-3xl font-bold" style={{ color: item.color }}>{item.value}</div>
              <div className="text-gray-600">{item.name}</div>
              <div className="text-sm text-gray-500">{item.percentage}%</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
