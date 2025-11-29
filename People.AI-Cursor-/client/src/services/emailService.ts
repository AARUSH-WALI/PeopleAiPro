import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

interface SendPersonalityTestEmailParams {
  email: string;
  name?: string;
  testLink: string;
}

interface EmailResponse {
  success: boolean;
  message?: string;
  error?: string;
  data?: {
    messageId: string;
    email: string;
  };
}

/**
 * Send personality test invitation email to candidate
 * @param params - Email parameters
 * @returns Promise with success status
 */
export async function sendPersonalityTestEmail(
  params: SendPersonalityTestEmailParams
): Promise<EmailResponse> {
  try {
    const { email, name, testLink } = params;

    if (!email || !email.trim()) {
      return {
        success: false,
        error: 'Email is required'
      };
    }

    if (!testLink || !testLink.trim()) {
      return {
        success: false,
        error: 'Test link is required'
      };
    }

    const response = await axios.post(`${API_BASE_URL}/email/send-personality-test`, {
      email: email.trim(),
      name: name || undefined,
      testLink: testLink.trim()
    });

    return {
      success: true,
      message: response.data.message,
      data: response.data.data
    };
  } catch (error: any) {
    console.error('Error sending personality test email:', error);
    return {
      success: false,
      error: error.response?.data?.message || error.message || 'Failed to send email'
    };
  }
}

