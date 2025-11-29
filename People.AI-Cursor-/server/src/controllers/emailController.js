import { sendPersonalityTestEmail } from '../services/emailService.js';

/**
 * Send personality test invitation email
 * POST /api/email/send-personality-test
 */
export const sendPersonalityTestInvite = async (req, res) => {
  try {
    const { email, name, testLink } = req.body;

    // Validate required fields
    if (!email || !email.trim()) {
      return res.status(400).json({
        success: false,
        error: 'Email is required',
        message: 'Recipient email address is required'
      });
    }

    if (!testLink || !testLink.trim()) {
      return res.status(400).json({
        success: false,
        error: 'Test link is required',
        message: 'Personality test link URL is required'
      });
    }

    // Validate email format (basic)
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email.trim())) {
      return res.status(400).json({
        success: false,
        error: 'Invalid email format',
        message: 'Please provide a valid email address'
      });
    }

    console.log('Sending personality test invitation:', {
      email: email.trim(),
      name: name || 'Candidate',
      testLink: testLink.trim()
    });

    // Send email
    const result = await sendPersonalityTestEmail(
      email.trim(),
      name || 'Candidate',
      testLink.trim()
    );

    if (!result.success) {
      return res.status(500).json({
        success: false,
        error: result.error || 'Failed to send email',
        message: result.error || 'An error occurred while sending the email'
      });
    }

    res.status(200).json({
      success: true,
      message: 'Personality test invitation sent successfully',
      data: {
        messageId: result.messageId,
        email: email.trim()
      }
    });
  } catch (error) {
    console.error('Error in sendPersonalityTestInvite:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      message: error.message || 'An unexpected error occurred'
    });
  }
};

