
import React, { useEffect, useState } from 'react';

interface FitmentScoreGaugeProps {
  score: number;
}

export default function FitmentScoreGauge({ score }: FitmentScoreGaugeProps) {
  const [currentRotation, setCurrentRotation] = useState(0);
  const targetRotation = (score / 100) * 180; // 180 degrees for half circle

  // Determine color based on score
  const getColor = () => {
    if (score >= 70) return '#4ade80'; // green
    if (score >= 50) return '#facc15'; // yellow
    return '#ef4444'; // red
  };

  // Animate the gauge needle
  useEffect(() => {
    const animationDuration = 1500; // 1.5 seconds
    const startTime = performance.now();
    const startRotation = currentRotation;

    const animate = (currentTime: number) => {
      const elapsedTime = currentTime - startTime;
      const progress = Math.min(elapsedTime / animationDuration, 1);

      // Easing function for smoother animation
      const easeOutQuad = (t: number) => 1 - (1 - t) * (1 - t);
      const easedProgress = easeOutQuad(progress);

      const newRotation = startRotation + (targetRotation - startRotation) * easedProgress;
      setCurrentRotation(newRotation);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }, [score, targetRotation]);

  return (
    <div className="relative w-full mx-auto">
      {/* Gauge background and meter */}
      <svg viewBox="0 0 200 140" className="w-full drop-shadow-md">
        {/* Gauge background */}
        <path
          d="M 10 110 A 90 90 0 0 1 190 110"
          fill="none"
          stroke="#e5e7eb"
          strokeWidth="20"
          strokeLinecap="round"
        />

        {/* Red section (0-50%) */}
        <path
          d="M 10 110 A 90 90 0 0 1 55 29"
          fill="none"
          stroke="#ef4444"
          strokeWidth="20"
          strokeLinecap="round"
        />
        {/* Yellow section (50-70%) */}
        <path
          d="M 55 29 A 90 90 0 0 1 100 20"
          fill="none"
          stroke="#facc15"
          strokeWidth="20"
          strokeLinecap="round"
        />
        {/* Green section (70-100%) */}
        <path
          d="M 100 20 A 90 90 0 0 1 190 110"
          fill="none"
          stroke="#4ade80"
          strokeWidth="20"
          strokeLinecap="round"
        />

        {/* Score indicators */}
        <text x="20" y="130" textAnchor="middle" fontSize="12" fontWeight="medium" fill="#666">0</text>
        <text x="60" y="32" textAnchor="middle" fontSize="12" fontWeight="medium" fill="#666">50</text>
        <text x="115" y="24" textAnchor="middle" fontSize="12" fontWeight="medium" fill="#666">70</text>
        <text x="185" y="130" textAnchor="middle" fontSize="12" fontWeight="medium" fill="#666">100</text>

        {/* Gauge needle pivot point */}
        <circle cx="100" cy="110" r="8" fill="#374151" />

        {/* Gauge needle */}
        <g transform={`rotate(${currentRotation - 90} 100 110)`}>
          <line
            x1="100"
            y1="110"
            x2="100"
            y2="30"
            stroke="#1f2937"
            strokeWidth="4"
            strokeLinecap="round"
          />
          <circle cx="100" cy="30" r="6" fill={getColor()} stroke="#1f2937" strokeWidth="1.5" />
        </g>

        {/* Score text */}
        <text
          x="100"
          y="85"
          textAnchor="middle"
          fontSize="28"
          fontWeight="bold"
          fill="#374151"
        >
          {score.toFixed(0)}%
        </text>
      </svg>

      {/* Gauge labels aligned roughly with color regions */}
      <div className="mt-2 px-6 text-sm text-gray-600 font-medium flex justify-between">
        <span className="w-1/3 text-left">Poor</span>
        <span className="w-1/3 text-center">Average</span>
        <span className="w-1/3 text-right">Excellent</span>
      </div>
    </div>
  );
}
