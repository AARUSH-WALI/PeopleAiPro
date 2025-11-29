import React from 'react';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';
import { Radar } from 'react-chartjs-2';

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

interface PersonalityChartProps {
  scores: number[];
}

export function PersonalityChart({ scores }: PersonalityChartProps) {
  const data = {
    labels: ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness'],
    datasets: [
      {
        label: 'Your Personality Profile',
        data: scores.map(score => (score + 50) / 100),
        backgroundColor: 'rgba(16, 185, 129, 0.2)',
        borderColor: 'rgba(16, 185, 129, 1)',
        borderWidth: 2,
        pointBackgroundColor: 'rgba(16, 185, 129, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(16, 185, 129, 1)',
      },
    ],
  };

  const options = {
    scales: {
      r: {
        angleLines: {
          display: true,
          color: 'rgba(217, 119, 6, 0.1)',
        },
        grid: {
          color: 'rgba(217, 119, 6, 0.1)',
        },
        suggestedMin: 0,
        suggestedMax: 1,
        ticks: {
          stepSize: 0.2,
          callback: (value: number) => `${Math.round(value * 100)}%`,
          color: 'rgb(217, 119, 6)',
        },
        pointLabels: {
          color: 'rgb(120, 53, 15)',
          font: {
            weight: '600',
          },
        },
      },
    },
    plugins: {
      legend: {
        display: false,
      },
    },
  };

  return (
    <div className="aspect-square w-full max-w-md mx-auto mb-8">
      <Radar data={data} options={options} />
    </div>
  );
}