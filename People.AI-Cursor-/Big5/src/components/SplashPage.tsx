import React from 'react';
import { PlayCircle } from 'lucide-react';

interface SplashPageProps {
  onStart: () => void;
  isEmbedded?: boolean;
}

export function SplashPage({ onStart, isEmbedded = false }: SplashPageProps) {
  const traits = [
    {
      name: 'Openness',
      description: 'Appreciation for art, emotion, adventure, unusual ideas, curiosity, and variety of experience.',
      icon: '🎨'
    },
    {
      name: 'Conscientiousness',
      description: 'Self-discipline, organized, and achievement-oriented behavior.',
      icon: '📋'
    },
    {
      name: 'Extraversion',
      description: 'Energy, positive emotions, assertiveness, sociability, and stimulation in the company of others.',
      icon: '🌟'
    },
    {
      name: 'Agreeableness',
      description: 'Compassion, cooperation, and concern for others\' needs.',
      icon: '🤝'
    },
    {
      name: 'Neuroticism',
      description: 'Tendency to experience emotional instability, anxiety, moodiness, and irritability.',
      icon: '🌊'
    }
  ];

  return (
    <div className={`flex flex-col items-center justify-center ${isEmbedded ? 'min-h-[500px]' : 'min-h-screen'} bg-gradient-to-br from-amber-50 to-emerald-50`}>
      <div className="max-w-4xl mx-auto px-4 py-16 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h1 className={`${isEmbedded ? 'text-2xl' : 'text-5xl'} font-bold text-amber-900 mb-4`}>
            Big Five Personality Test
          </h1>
          <p className={`${isEmbedded ? 'text-sm' : 'text-xl'} text-amber-800`}>
            Discover your personality traits through scientifically validated assessment
          </p>
        </div>

        {!isEmbedded && (
          <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3 mb-12">
            {traits.map((trait) => (
              <div
                key={trait.name}
                className="bg-white/90 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-amber-100"
              >
                <div className="text-4xl mb-4">{trait.icon}</div>
                <h3 className="text-xl font-semibold text-amber-900 mb-2">
                  {trait.name}
                </h3>
                <p className="text-amber-700">
                  {trait.description}
                </p>
              </div>
            ))}
          </div>
        )}

        <div className="bg-white/90 backdrop-blur-sm rounded-xl p-8 shadow-lg border border-amber-100 text-center">
          <h2 className={`${isEmbedded ? 'text-lg' : 'text-2xl'} font-bold text-amber-900 mb-4`}>
            Ready to Begin?
          </h2>
          {!isEmbedded && (
            <p className="text-amber-700 mb-6">
              This test consists of 50 questions and takes about 10 minutes to complete.
              Your responses will help you understand your personality across five fundamental dimensions.
            </p>
          )}
          <button
            onClick={onStart}
            className="inline-flex items-center gap-2 px-6 py-3 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 transition-colors"
          >
            <PlayCircle size={24} />
            Start Test
          </button>
        </div>
      </div>
    </div>
  );
}