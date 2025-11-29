// Routes for retention analysis operations
import express from 'express';
import { backfillRetentionAnalysis } from '../scripts/backfillRetention.js';
import { backfillDatasetScore } from '../scripts/backfillDatasetScore.js';

const router = express.Router();

/**
 * POST /api/retention/backfill
 * Backfill retention analysis for all existing candidates with personality data
 */
router.post('/backfill', async (req, res) => {
    try {
        console.log('Retention backfill requested');
        const result = await backfillRetentionAnalysis();
        res.json({
            success: true,
            message: 'Retention analysis backfill completed',
            ...result
        });
    } catch (error) {
        console.error('Error in retention backfill:', error);
        res.status(500).json({
            success: false,
            error: 'Backfill failed',
            message: error.message
        });
    }
});

/**
 * POST /api/retention/backfill-dataset-score
 * Backfill dataset_score for all existing candidates
 */
router.post('/backfill-dataset-score', async (req, res) => {
    try {
        console.log('Dataset score backfill requested');
        const result = await backfillDatasetScore();
        res.json({
            success: true,
            message: 'Dataset score backfill completed',
            ...result
        });
    } catch (error) {
        console.error('Error in dataset score backfill:', error);
        res.status(500).json({
            success: false,
            error: 'Backfill failed',
            message: error.message
        });
    }
});

export default router;

