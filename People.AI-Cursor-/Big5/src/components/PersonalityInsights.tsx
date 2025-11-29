import { useEffect, useState } from 'react';
import { analyzePersonality } from '../utils/gemini';
import { questions } from '../data/questions';

interface PersonalityInsightsProps {
  scores: number[];
  traitScores: {
    openness: number;
    conscientiousness: number;
    extraversion: number;
    agreeableness: number;
    neuroticism: number;
  };
}

export function PersonalityInsights({ scores, traitScores }: PersonalityInsightsProps) {
  const [analysis, setAnalysis] = useState<string>('');
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const getAnalysis = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const answers = scores.map((score, index) => ({
          question: questions[index].text,
          score
        }));

        const result = await analyzePersonality(answers, traitScores);
        setAnalysis(result.analysis);
        setRecommendations(result.recommendations);
      } catch (err) {
        setError('Failed to generate personality insights. Please try again.');
        console.error('Analysis error:', err);
      } finally {
        setLoading(false);
      }
    };

    getAnalysis();
  }, [scores, traitScores]);

  if (error) {
    return (
      <div className="mt-8 p-4 bg-red-50 rounded-lg border border-red-200">
        <p className="text-red-600">{error}</p>
      </div>
    );
  }

  return (
    <div className="mt-8 bg-white/95 backdrop-blur-sm rounded-xl shadow-lg p-6 border border-amber-100">
      <h3 className="text-xl font-bold text-amber-900 mb-4">
        AI-Powered Personality Analysis
      </h3>

      {loading ? (
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-amber-100 rounded w-3/4"></div>
          <div className="h-4 bg-amber-100 rounded w-5/6"></div>
          <div className="h-4 bg-amber-100 rounded w-2/3"></div>
        </div>
      ) : (
        <>
          <div className="prose prose-amber max-w-none">
            <p className="text-amber-800 leading-relaxed">{analysis}</p>
          </div>

          {recommendations.length > 0 && (
            <>
              <h4 className="text-lg font-semibold text-amber-900 mt-6 mb-3">
                Personal Growth Recommendations
              </h4>
              <ul className="list-disc list-inside space-y-2">
                {recommendations.map((rec, index) => (
                  <li key={index} className="text-amber-800">{rec}</li>
                ))}
              </ul>
            </>
          )}
        </>
      )}
    </div>
  );
}
