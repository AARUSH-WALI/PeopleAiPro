// // Fitment scoring service ported from provided Python logic

// /**
//  * Normalize Big5 scores from app scale (-50..50) to model scale (0..40)
//  */
// const normalizeBig5ForModel = (val) => {
//     if (val === null || val === undefined || isNaN(Number(val))) return 0;
//     const clamped = Math.max(-50, Math.min(50, Number(val)));
//     return Math.round(((clamped + 50) / 100) * 40);
// };

// export const categorizeCandidate = (longevityYears, averageExperience) => {
//     const ly = Number(longevityYears ?? 0);
//     const ae = Number(averageExperience ?? 0);
//     if (ly >= 5 && ae >= 3) return 'Experienced';
//     if (ly > 1 && ae > 1) return 'Inexperienced';
//     return 'Fresher';
// };

// const scoreLongevity = (val) => (val >= 6 ? 100 : val >= 4 ? 75 : val >= 1 ? 50 : 20);
// const scoreAvgExp = (val) => (val >= 3 ? 100 : val >= 1.8 ? 75 : val >= 1 ? 50 : 20);
// const scoreWorkshops = (val) => (val >= 12.73 ? 100 : val >= 7 ? 75 : val >= 3 ? 50 : 20);
// const scoreTrainings = (val) => (val >= 12.90 ? 100 : val >= 7 ? 75 : val >= 3 ? 50 : 20);
// const scorePapers = (val) => (val >= 1.18 ? 100 : val >= 0.5 ? 75 : val >= 0.2 ? 50 : 0);
// const scorePatents = (val) => (val >= 0.04 ? 100 : 0);
// const scoreAchievements = (val) => (val >= 7.54 ? 100 : val >= 4 ? 75 : val >= 1 ? 50 : 0);
// const scoreBooks = (val) => (val >= 0.81 ? 100 : 0);
// const scoreState = (val) => (val === 1 ? 100 : 0);
// const scoreJobs = (val) => (val >= 0.17 ? 100 : 0);
// const scoreInstitute = (val) => (val === 1 ? 100 : 0);

// export const calculateDatasetScore = (data) => {
//     const scores = {};

//     // Direct scores (no normalization needed)
//     scores.longevity = scoreLongevity(Number(data.longevity_years || 0)) * 0.30;
//     scores.avg_exp = scoreAvgExp(Number(data.average_experience || 0)) * 0.30;
//     scores.state = scoreState((data.is_jk ? 1 : 0)) * 0.02;

//     // Jobs score - normalize by experience (jobs per year)
//     const experienceYears = Math.max(1, Number(data.average_experience || data.longevity_years || 1));
//     const jobsRate = Number(data.number_of_jobs || 0) / experienceYears;
//     scores.jobs = scoreJobs(jobsRate) * 0.01;

//     scores.ug = scoreInstitute(data.ug_institute ? 1 : 0) * 0.02;
//     scores.pg = scoreInstitute(data.pg_institute ? 1 : 0) * 0.03;
//     scores.phd = scoreInstitute(data.phd_institute ? 1 : 0) * 0.05;

//     // Rate-based scores (normalize by experience)
//     // Calculate rates (count per year of experience) - experienceYears already calculated above
//     const workshopsRate = Number(data.workshops_count || 0) / experienceYears;
//     const trainingsRate = Number(data.trainings_count || 0) / experienceYears;
//     const papersRate = Number(data.research_papers_count || 0) / experienceYears;
//     const patentsRate = Number(data.patents_count || 0) / experienceYears;
//     const achievementsRate = Number(data.achievements_count || 0) / experienceYears;
//     const booksRate = Number(data.books_count || 0) / experienceYears;

//     // Score based on rates
//     scores.workshops = scoreWorkshops(workshopsRate) * 0.045;
//     scores.trainings = scoreTrainings(trainingsRate) * 0.045;
//     scores.papers = scorePapers(papersRate) * 0.05;
//     scores.patents = scorePatents(patentsRate) * 0.07;
//     scores.achievements = scoreAchievements(achievementsRate) * 0.04;
//     scores.books = scoreBooks(booksRate) * 0.02;

//     // Sum all scores (should be 0-100)
//     const totalScore = Object.values(scores).reduce((a, b) => a + b, 0);

//     // Ensure score is between 0 and 100
//     return Math.max(0, Math.min(100, totalScore));
// };

// export const scaleDatasetScore = (rawScore, category) => {
//     // Experienced: dataset 70, Big5 30; Fresher/Inexperienced: dataset 30, Big5 70
//     return category === 'Experienced' ? (rawScore / 100) * 70 : (rawScore / 100) * 30;
// };

// export const calculateBig5Score = (big5Data, category) => {
//     const O = normalizeBig5ForModel(big5Data?.openness);
//     const C = normalizeBig5ForModel(big5Data?.conscientiousness);
//     const E = normalizeBig5ForModel(big5Data?.extraversion);
//     const A = normalizeBig5ForModel(big5Data?.agreeableness);
//     const N = normalizeBig5ForModel(big5Data?.neuroticism);

//     const toBucket = (x, buckets) => {
//         if (x <= 10) return buckets[0];
//         if (x <= 20) return buckets[1];
//         if (x <= 30) return buckets[2];
//         return buckets[3];
//     };

//     const scores = {
//         O: toBucket(O, [0.25, 0.50, 0.75, 1.0]),
//         C: toBucket(C, [0.25, 0.50, 0.75, 1.0]),
//         E: toBucket(E, [0.50, 0.75, 1.0, 0.75]),
//         A: toBucket(A, [0.25, 0.50, 1.0, 0.75]),
//         N: toBucket(N, [1.0, 0.75, 0.50, 0.25])
//     };

//     // Big5 weight: 30 for Experienced, 70 for Fresher/Inexperienced
//     const weightPerTrait = (category === 'Experienced' ? 30 : 70) / 5;
//     const total = Object.values(scores).reduce((sum, v) => sum + v * weightPerTrait, 0);
//     return { trait_scores: scores, total_score: total, weight_per_trait: weightPerTrait };
// };

// export const calculateOverallFitment = (candidateData, big5Data) => {
//     const category = categorizeCandidate(candidateData.longevity_years || 0, candidateData.average_experience || 0);
//     const rawDataset = calculateDatasetScore(candidateData);
//     const datasetScaled = scaleDatasetScore(rawDataset, category);
//     const big5 = calculateBig5Score(big5Data || {}, category);
//     const overall = datasetScaled + big5.total_score;
//     return {
//         category,
//         raw_dataset_score: Math.round(rawDataset * 100) / 100,
//         fitment_score: Math.round(datasetScaled * 100) / 100,
//         big5_score: Math.round(big5.total_score * 100) / 100,
//         big5_trait_scores: big5.trait_scores,
//         overall_fitment_score: Math.round(overall * 100) / 100
//     };
// };



// ----------------------------------------------
// EXACT PYTHON → JS REWRITE OF FITMENT MODEL
// ----------------------------------------------

// Normalize Big Five from -50..50 to 0..40
const normalizeBig5ForModel = (val) => {
  if (val === null || val === undefined || isNaN(Number(val))) return 0;
  const clamped = Math.max(-50, Math.min(50, Number(val)));
  return ((clamped + 50) / 100) * 40;
};

// Categorize candidate to align with EnhancedFitmentScoringSystem
// Use longevity years only, and switch to technical-heavy weighting at 3+ years
export const categorizeCandidate = (longevityYears, averageExperience) => {
  const ly = Number(longevityYears ?? 0);

  if (ly >= 3) return "Experienced";   // 3+ years: technical-heavy (70% technical, 30% personality)
  if (ly > 1) return "Inexperienced"; // 1-3 years: still early-career but not fresher
  return "Fresher";                   // <=1 year: fresher, personality-heavy
};

// Scoring helpers (Python-equivalent thresholds)
const scoreLongevity = (v) => (v >= 6 ? 100 : v >= 4 ? 75 : v >= 1 ? 50 : 20);
const scoreAvgExp = (v) => (v >= 3 ? 100 : v >= 1.8 ? 75 : v >= 1 ? 50 : 20);
const scoreWorkshops = (v) => (v >= 12.73 ? 100 : v >= 7 ? 75 : v >= 3 ? 50 : 20);
const scoreTrainings = (v) => (v >= 12.90 ? 100 : v >= 7 ? 75 : v >= 3 ? 50 : 20);
const scorePapers = (v) => (v >= 1.18 ? 100 : v >= 0.5 ? 75 : v >= 0.2 ? 50 : 0);
const scorePatents = (v) => (v >= 0.04 ? 100 : 0);
const scoreAchievements = (v) => (v >= 7.54 ? 100 : v >= 4 ? 75 : v >= 1 ? 50 : 0);
const scoreBooks = (v) => (v >= 0.81 ? 100 : 0);
const scoreState = (v) => (v === 1 ? 100 : 0);
const scoreJobs = (v) => (v >= 0.17 ? 100 : 0);
const scoreInstitute = (v) => (v === 1 ? 100 : 0);

// -----------------------------
// PYTHON-DERIVED DATASET SCORE
// -----------------------------
export const calculateDatasetScore = (data) => {
  const scores = {};

  const longevity = Number(data.longevity_years || 0);
  const avgExp = Number(data.average_experience || 0);
  const isJK = data.is_jk ? 1 : 0;
  const ug = data.ug_institute ? 1 : 0;
  const pg = data.pg_institute ? 1 : 0;
  const phd = data.phd_institute ? 1 : 0;
  const numJobs = Number(data.number_of_jobs || 0);

  const expYears = Math.max(
    1,
    Number(data.average_experience || data.longevity_years || 1)
  );

  // Rate-based fields
  const workshopRate = Number(data.workshops_count || 0) / expYears;
  const trainingRate = Number(data.trainings_count || 0) / expYears;
  const papersRate = Number(data.research_papers_count || 0) / expYears;
  const patentsRate = Number(data.patents_count || 0) / expYears;
  const achievementsRate = Number(data.achievements_count || 0) / expYears;
  const booksRate = Number(data.books_count || 0) / expYears;

  // Assign all weighted scores (matching Python)
  scores.longevity = scoreLongevity(longevity) * 0.30;
  scores.avg_exp = scoreAvgExp(avgExp) * 0.30;
  scores.state = scoreState(isJK) * 0.02;

  scores.jobs = scoreJobs(numJobs / expYears) * 0.01;

  scores.ug = scoreInstitute(ug) * 0.02;
  scores.pg = scoreInstitute(pg) * 0.03;
  scores.phd = scoreInstitute(phd) * 0.05;

  scores.workshops = scoreWorkshops(workshopRate) * 0.045;
  scores.trainings = scoreTrainings(trainingRate) * 0.045;
  scores.papers = scorePapers(papersRate) * 0.05;
  scores.patents = scorePatents(patentsRate) * 0.07;
  scores.achievements = scoreAchievements(achievementsRate) * 0.04;
  scores.books = scoreBooks(booksRate) * 0.02;

  const total =
    Object.values(scores).reduce((a, b) => a + b, 0);

  return Math.min(100, Math.max(0, total));
};

// -------------------------------
// SCALE DATASET SCORE (same as Python)
// -------------------------------
export const scaleDatasetScore = (rawScore, category) =>
  category === "Experienced"
    ? (rawScore / 100) * 70
    : (rawScore / 100) * 30;

// -------------------------------------
// BIG FIVE TRAIT CALCULATION (MATCHES PYTHON)
// -------------------------------------
export const calculateBig5Score = (big5Data, category) => {
  const O = normalizeBig5ForModel(big5Data?.openness);
  const C = normalizeBig5ForModel(big5Data?.conscientiousness);
  const E = normalizeBig5ForModel(big5Data?.extraversion);
  const A = normalizeBig5ForModel(big5Data?.agreeableness);
  const N = normalizeBig5ForModel(big5Data?.neuroticism);

  const toBucket = (x, buckets) => {
    if (x <= 10) return buckets[0];
    if (x <= 20) return buckets[1];
    if (x <= 30) return buckets[2];
    return buckets[3];
  };

  const scores = {
    O: toBucket(O, [0.25, 0.50, 0.75, 1.0]),
    C: toBucket(C, [0.25, 0.50, 0.75, 1.0]),
    E: toBucket(E, [0.50, 0.75, 1.0, 0.75]),
    A: toBucket(A, [0.25, 0.50, 1.0, 0.75]),
    N: toBucket(N, [1.0, 0.75, 0.50, 0.25]),
  };

  const weight = category === "Experienced" ? 30 : 70;
  const perTraitWeight = weight / 5;

  const total = Object.values(scores).reduce(
    (sum, v) => sum + v * perTraitWeight,
    0
  );

  return {
    trait_scores: scores,
    total_score: total,
    weight_per_trait: perTraitWeight,
  };
};

// -----------------------------
// FINAL FITMENT SCORE
// -----------------------------
export const calculateOverallFitment = (candidateData, big5Data) => {
  const category = categorizeCandidate(
    candidateData.longevity_years || 0,
    candidateData.average_experience || 0
  );

  const rawDataset = calculateDatasetScore(candidateData);
  const datasetScaled = scaleDatasetScore(rawDataset, category);

  const big5 = calculateBig5Score(big5Data || {}, category);

  const overall = datasetScaled + big5.total_score;

  return {
    category,
    raw_dataset_score: Math.round(rawDataset * 100) / 100,
    fitment_score: Math.round(datasetScaled * 100) / 100,
    big5_score: Math.round(big5.total_score * 100) / 100,
    big5_trait_scores: big5.trait_scores,
    overall_fitment_score: Math.round(overall * 100) / 100,
  };
};

export default {
  categorizeCandidate,
  calculateDatasetScore,
  calculateBig5Score,
  calculateOverallFitment,
};
