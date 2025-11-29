import React, { useState, useEffect } from 'react';
import { Dice6, ArrowLeft, ArrowRight, PlayCircle } from 'lucide-react';
import { questions } from './data/questions';
import { ResultsCard } from './components/ResultsCard';
import { SplashPage } from './components/SplashPage';

interface AppProps {
  isEmbedded?: boolean;
}

function App({ isEmbedded = false }: AppProps) {
  const [started, setStarted] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [scores, setScores] = useState<number[]>(new Array(questions.length).fill(0));
  const [isComplete, setIsComplete] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);
  const [intermediateScores, setIntermediateScores] = useState<Array<{
    code: number;
    originalScore: number;
    finalScore: number;
  }>>([]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setIsAdmin(params.get('admin') === 'true');
  }, []);

  const handleStart = () => setStarted(true);

  const handleNext = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(curr => curr + 1);
    }
  };

  const handlePrev = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(curr => curr - 1);
    }
  };

  const handleScore = (originalScore: number) => {
    const newScores = [...scores];
    const finalScore = questions[currentQuestion].code < 0 ? (6 - originalScore) : originalScore;
    newScores[currentQuestion] = finalScore;
    setScores(newScores);
  };

  const handleComplete = () => {
    const intermediateResults = scores.map((score, index) => ({
      code: questions[index].code,
      originalScore: questions[index].code < 0 ? 6 - score : score,
      finalScore: score
    }));
    setIntermediateScores(intermediateResults);
    setIsComplete(true);
  };

  const containerClasses = isEmbedded 
    ? 'min-h-screen bg-gradient-to-br from-amber-50 to-amber-100 font-sans text-amber-900 p-4'
    : 'container mx-auto max-w-2xl min-h-screen bg-gradient-to-br from-amber-50 to-amber-100 font-sans text-amber-900 p-4';

  return (
    <div className={containerClasses}>
      {!started ? (
        <SplashPage onStart={handleStart} isEmbedded={isEmbedded} />
      ) : !isComplete ? (
        <div className="mt-8">
          <div className="flex justify-between items-center mb-8">
            <button
              onClick={handlePrev}
              disabled={currentQuestion === 0}
              className="p-2 text-amber-600 disabled:text-amber-300 hover:text-amber-700 disabled:hover:text-amber-300"
            >
              <ArrowLeft size={24} />
            </button>
            <span className="text-lg font-medium">
              Question {currentQuestion + 1} of {questions.length}
            </span>
            {currentQuestion === questions.length - 1 ? (
              <button
                onClick={handleComplete}
                className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700"
              >
                <PlayCircle size={20} />
                Complete
              </button>
            ) : (
              <button
                onClick={handleNext}
                disabled={currentQuestion === questions.length - 1}
                className="p-2 text-amber-600 disabled:text-amber-300 hover:text-amber-700 disabled:hover:text-amber-300"
              >
                <ArrowRight size={24} />
              </button>
            )}
          </div>

          <div className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl p-8 mb-8 border border-amber-100">
            <h2 className="text-xl font-bold mb-6">{questions[currentQuestion].text}</h2>
            <div className="grid grid-cols-5 gap-4">
              {[1, 2, 3, 4, 5].map((score) => (
                <button
                  key={score}
                  onClick={() => {
                    handleScore(score);
                    if (currentQuestion < questions.length - 1) {
                      handleNext();
                    }
                  }}
                  className={`p-4 rounded-xl font-medium transition-all ${
                    scores[currentQuestion] === (questions[currentQuestion].code < 0 ? 6 - score : score)
                      ? 'bg-emerald-600 text-white'
                      : 'bg-amber-100 hover:bg-amber-200 text-amber-900'
                  }`}
                >
                  {score}
                </button>
              ))}
            </div>
            <div className="flex justify-between text-sm text-amber-600 mt-4">
              <span>Strongly Disagree</span>
              <span>Strongly Agree</span>
            </div>
          </div>

          {isAdmin && (
            <div className="space-y-4 text-sm font-mono">
              {intermediateScores.map((score, index) => (
                <div key={index}>
                  Q{index + 1} (Code: {score.code}): Original: {score.originalScore}, Final:{' '}
                  {score.finalScore}
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <ResultsCard scores={scores} />
      )}
    </div>
  );
}

export default App;