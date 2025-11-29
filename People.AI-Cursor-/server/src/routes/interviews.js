import express from 'express';
import { getInterviews, getInterviewById, createInterview, updateInterview, deleteInterview, updateInterviewStatus } from '../controllers/interviewController.js';

const router = express.Router();

// GET /api/interviews - Get all interviews with pagination and filtering
router.get('/', getInterviews);

// GET /api/interviews/:id - Get a specific interview by ID
router.get('/:id', getInterviewById);

// POST /api/interviews - Create a new interview
router.post('/', createInterview);

// PUT /api/interviews/:id - Update an interview
router.put('/:id', updateInterview);

// DELETE /api/interviews/:id - Delete an interview
router.delete('/:id', deleteInterview);

// PATCH /api/interviews/:id/status - Update interview status
router.patch('/:id/status', updateInterviewStatus);

export default router;
