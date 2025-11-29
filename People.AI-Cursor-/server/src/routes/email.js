import express from 'express';
import { sendPersonalityTestInvite } from '../controllers/emailController.js';

const router = express.Router();

// POST /api/email/send-personality-test - Send personality test invitation email
router.post('/send-personality-test', sendPersonalityTestInvite);

export default router;

