
import React, { useEffect, useState } from "react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";

interface PersonalityScores {
  extraversion: number;
  agreeableness: number;
  openness: number;
  neuroticism: number;
  conscientiousness: number;
}

interface PersonalityPieChartProps {
  scores: PersonalityScores;
}

export default function PersonalityPieChart({ scores }: PersonalityPieChartProps) {
  const baseData = [
    { name: "Extroversion", value: scores.extraversion, color: "#8B5CF6" }, // Purple
    { name: "Agreeableness", value: scores.agreeableness, color: "#10B981" }, // Green
    { name: "Openness", value: scores.openness, color: "#3B82F6" }, // Blue
    { name: "Neuroticism", value: scores.neuroticism, color: "#F59E0B" }, // Amber
    { name: "Conscientiousness", value: scores.conscientiousness, color: "#EC4899" }, // Pink
  ];

  const [animatedData, setAnimatedData] = useState(
    baseData.map((d) => ({ ...d, value: 0 }))
  );

  useEffect(() => {
    // Animate slice values from 0 up to their actual scores
    const start = performance.now();
    const duration = 900;

    const animate = (time: number) => {
      const t = Math.min((time - start) / duration, 1);
      const easeOutCubic = (x: number) => 1 - Math.pow(1 - x, 3);
      const eased = easeOutCubic(t);

      setAnimatedData(
        baseData.map((d) => ({
          ...d,
          value: d.value * eased,
        }))
      );

      if (t < 1) requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  }, [scores.extraversion, scores.agreeableness, scores.openness, scores.neuroticism, scores.conscientiousness]);

  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={animatedData}
            cx="50%"
            cy="50%"
            labelLine={false}
            outerRadius={120}
            innerRadius={60}
            fill="#8884d8"
            dataKey="value"
            label={({ value }) => `${value}%`}
            paddingAngle={2}
            isAnimationActive={true}
            animationBegin={0}
            animationDuration={800}
            animationEasing="ease-out"
          >
            {animatedData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} strokeWidth={2} />
            ))}
          </Pie>
          <Tooltip
            formatter={(value) => `${value}%`}
            contentStyle={{ borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}
            itemStyle={{ color: "#333" }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
