import express from 'express';
import {
    getJobs,
    getJobById,
    createJob,
    updateJob,
    deleteJob
} from '../controllers/jobController.js';

const router = express.Router();

// GET /api/jobs - Get all jobs
router.get('/', getJobs);

// POST /api/jobs - Create a new job
router.post('/', createJob);

// GET /api/jobs/:id - Get a single job by ID
router.get('/:id', getJobById);

// PUT /api/jobs/:id - Update a job (can update active status)
router.put('/:id', updateJob);

// DELETE /api/jobs/:id - Delete a job
router.delete('/:id', deleteJob);

export default router;

