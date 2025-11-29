import { questions } from '../data/questions';

// Using modulo 5 to determine trait index:
// 1 -> Extraversion
// 2 -> Agreeableness
// 3 -> Conscientiousness
// 4 -> Neuroticism
// 5 -> Openness

export function calculateTraitScores(scores: number[]): number[] {
  const traitScores = new Array(5).fill(0);

  scores.forEach((score, index) => {
    const question = questions[index];
    const traitIndex = (Math.abs(question.code) % 5 || 5) - 1;
    traitScores[traitIndex] += score;
  });

  return traitScores;
}