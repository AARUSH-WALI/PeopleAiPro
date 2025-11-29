// Script to backfill retention analysis for existing candidates
// This calculates retention scores for candidates who already have personality test data
// but don't have retention analysis yet

import { supabase } from '../config/supabase.js';
import { calculateOverallFitment } from '../services/fitmentService.js';
import { retentionScorer } from '../services/retentionService.js';

/**
 * Backfill retention analysis for all candidates with personality data
 */
export async function backfillRetentionAnalysis() {
    try {
        console.log('Starting retention analysis backfill...');

        // Get all resumes that have personality test completed but no retention data
        const { data: candidates, error: fetchError } = await supabase
            .from('resumes')
            .select('*')
            .not('conscientiousness', 'is', null)
            .not('agreeableness', 'is', null)
            .not('extraversion', 'is', null)
            .not('openness', 'is', null)
            .not('neuroticism', 'is', null);

        if (fetchError) {
            throw new Error(`Error fetching candidates: ${fetchError.message}`);
        }

        if (!candidates || candidates.length === 0) {
            console.log('No candidates found with personality test data.');
            return { success: true, processed: 0, message: 'No candidates to process' };
        }

        console.log(`Found ${candidates.length} candidates with personality test data.`);

        let processed = 0;
        let errors = 0;
        const errorsList = [];

        for (const record of candidates) {
            try {
                console.log(`Processing candidate: ${record.email} (${record.name})`);

                // Recompute fitment first (needed for retention calculation)
                const fitment = calculateOverallFitment(
                    {
                        longevity_years: record.longevity_years,
                        average_experience: record.average_experience,
                        workshops_count: record.workshops_count,
                        trainings_count: record.trainings_count,
                        research_papers_count: record.research_papers_count,
                        patents_count: record.patents_count,
                        achievements_count: record.achievements_count,
                        books_count: record.books_count,
                        is_jk: record.is_jk,
                        number_of_jobs: record.number_of_jobs,
                        ug_institute: record.ug_institute,
                        pg_institute: record.pg_institute,
                        phd_institute: record.phd_institute
                    },
                    {
                        extraversion: record.extraversion,
                        agreeableness: record.agreeableness,
                        conscientiousness: record.conscientiousness,
                        neuroticism: record.neuroticism,
                        openness: record.openness
                    }
                );

                const overall = fitment.overall_fitment_score;
                const personalityScore = Math.round(fitment.big5_score);

                // Normalize candidate_type
                const allowedTypes = new Set(['Experienced', 'Intermediate', 'Fresher']);
                const normalizedType = typeof fitment.category === 'string'
                    ? (() => {
                        const t = fitment.category.trim().toLowerCase();
                        if (t === 'experienced') return 'Experienced';
                        if (t === 'intermediate' || t === 'inexperienced') return 'Intermediate';
                        if (t === 'fresher') return 'Fresher';
                        return 'Fresher';
                    })()
                    : 'Fresher';
                const candidateType = allowedTypes.has(normalizedType) ? normalizedType : 'Fresher';

                // Calculate retention risk
                const candidateDataForRetention = {
                    ...record,
                    number_of_unique_designations: record.number_of_unique_designations ?? record.number_of_jobs,
                    workshops: record.workshops_count,
                    trainings: record.trainings_count,
                    total_papers: record.research_papers_count,
                    total_patents: record.patents_count,
                    achievements: record.achievements_count,
                    fitment_score: overall
                };

                const big5ScoresForRetention = {
                    conscientiousness: record.conscientiousness,
                    agreeableness: record.agreeableness,
                    neuroticism: record.neuroticism,
                    extraversion: record.extraversion,
                    openness: record.openness
                };

                const retentionAnalysis = retentionScorer.calculateRetentionRisk(
                    candidateDataForRetention,
                    overall,
                    big5ScoresForRetention,
                    candidateType
                );

                console.log(`  Retention Score: ${retentionAnalysis.retention_score}, Risk: ${retentionAnalysis.retention_risk}`);

                // Update the database
                const { error: updateError } = await supabase
                    .from('resumes')
                    .update({
                        fitment_score: overall,
                        profile_score: Math.round(overall),
                        personality_score: personalityScore,
                        candidate_type: candidateType,
                        retention_score: retentionAnalysis.retention_score,
                        retention_risk: retentionAnalysis.retention_risk,
                        retention_analysis: retentionAnalysis
                    })
                    .eq('id', record.id);

                if (updateError) {
                    throw new Error(`Database update failed: ${updateError.message}`);
                }

                processed++;
            } catch (error) {
                errors++;
                const errorMsg = `Error processing ${record.email}: ${error.message}`;
                console.error(errorMsg);
                errorsList.push(errorMsg);
            }
        }

        const result = {
            success: true,
            processed,
            errors,
            total: candidates.length,
            errorsList: errorsList.length > 0 ? errorsList : undefined
        };

        console.log('\n=== Backfill Complete ===');
        console.log(`Processed: ${processed}`);
        console.log(`Errors: ${errors}`);
        console.log(`Total: ${candidates.length}`);

        return result;
    } catch (error) {
        console.error('Fatal error in backfill:', error);
        throw error;
    }
}

// Function is already exported above as a named export

