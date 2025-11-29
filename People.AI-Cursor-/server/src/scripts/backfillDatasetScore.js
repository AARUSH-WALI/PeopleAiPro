// Script to backfill dataset_score for existing candidates
// This calculates and updates dataset_score for all resumes that don't have it

import { supabase } from '../config/supabase.js';
import { calculateDatasetScore } from '../services/fitmentService.js';

/**
 * Backfill dataset_score for all candidates missing this value
 */
export async function backfillDatasetScore() {
    try {
        console.log('Starting dataset_score backfill...');

        // Get all resumes that don't have dataset_score or have null dataset_score
        const { data: resumes, error: fetchError } = await supabase
            .from('resumes')
            .select('*')
            .or('dataset_score.is.null,dataset_score.eq.0');

        if (fetchError) {
            throw new Error(`Error fetching resumes: ${fetchError.message}`);
        }

        if (!resumes || resumes.length === 0) {
            console.log('No resumes found without dataset_score.');
            return { success: true, processed: 0, message: 'No resumes to process' };
        }

        console.log(`Found ${resumes.length} resumes without dataset_score.`);

        let processed = 0;
        let errors = 0;
        const errorsList = [];

        for (const record of resumes) {
            try {
                console.log(`Processing resume: ${record.email} (${record.name})`);

                // Prepare candidate data for dataset calculation
                const candidateDataForDataset = {
                    longevity_years: record.longevity_years || 0,
                    average_experience: record.average_experience || 0,
                    workshops_count: record.workshops_count || 0,
                    trainings_count: record.trainings_count || 0,
                    research_papers_count: record.research_papers_count || 0,
                    patents_count: record.patents_count || 0,
                    achievements_count: record.achievements_count || 0,
                    books_count: record.books_count || 0,
                    is_jk: record.is_jk || false,
                    number_of_jobs: record.number_of_jobs || 0,
                    ug_institute: record.ug_institute || null,
                    pg_institute: record.pg_institute || null,
                    phd_institute: record.phd_institute || null
                };

                // Calculate dataset score
                const datasetScore = calculateDatasetScore(candidateDataForDataset);
                const roundedScore = Math.round(datasetScore * 100) / 100;

                console.log(`  Dataset Score: ${roundedScore}/100`);

                // Update the database
                const { error: updateError } = await supabase
                    .from('resumes')
                    .update({
                        dataset_score: roundedScore
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
            total: resumes.length,
            errorsList: errorsList.length > 0 ? errorsList : undefined
        };

        console.log('\n=== Backfill Complete ===');
        console.log(`Processed: ${processed}`);
        console.log(`Errors: ${errors}`);
        console.log(`Total: ${resumes.length}`);

        return result;
    } catch (error) {
        console.error('Fatal error in backfill:', error);
        throw error;
    }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    backfillDatasetScore()
        .then(result => {
            console.log('\n✅ Backfill completed successfully');
            process.exit(0);
        })
        .catch(error => {
            console.error('\n❌ Backfill failed:', error);
            process.exit(1);
        });
}


