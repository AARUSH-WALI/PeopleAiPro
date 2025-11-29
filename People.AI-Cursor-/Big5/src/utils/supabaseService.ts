// Using server API endpoint instead of direct Supabase client for security
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

export interface PersonalityTestScores {
  extraversion: number;
  agreeableness: number;
  conscientiousness: number;
  neuroticism: number;
  openness: number;
}

/**
 * Saves Big5 personality test results to Supabase via server API
 * @param email - Candidate email to identify the resume record
 * @param scores - Object containing the 5 personality trait scores
 * @returns Promise with success status and message
 */
export async function savePersonalityTestResults(
  email: string,
  scores: PersonalityTestScores
): Promise<{ success: boolean; message: string; error?: string }> {
  if (!email || !email.trim()) {
    return {
      success: false,
      message: 'Email is required',
      error: 'Email parameter is missing or empty'
    };
  }

  try {
    // Validate scores are numbers
    const {
      extraversion,
      agreeableness,
      conscientiousness,
      neuroticism,
      openness
    } = scores;

    if (
      typeof extraversion !== 'number' ||
      typeof agreeableness !== 'number' ||
      typeof conscientiousness !== 'number' ||
      typeof neuroticism !== 'number' ||
      typeof openness !== 'number'
    ) {
      return {
        success: false,
        message: 'Invalid score format',
        error: 'All scores must be numbers'
      };
    }

    console.log('Sending personality test results to server:', {
      email,
      scores: {
        extraversion,
        agreeableness,
        conscientiousness,
        neuroticism,
        openness
      }
    });

    // Call server API endpoint to update personality test results
    const response = await fetch(`${API_BASE_URL}/resumes/personality-test`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email: email.trim(),
        extraversion: Math.round(extraversion),
        agreeableness: Math.round(agreeableness),
        conscientiousness: Math.round(conscientiousness),
        neuroticism: Math.round(neuroticism),
        openness: Math.round(openness)
      })
    });

    const data = await response.json();

    if (!response.ok) {
      return {
        success: false,
        message: data.message || 'Failed to save test results',
        error: data.error || `Server error: ${response.status}`
      };
    }

    console.log('Successfully saved personality test results:', data);
    return {
      success: true,
      message: data.message || 'Personality test results saved successfully'
    };
  } catch (error) {
    console.error('Unexpected error saving personality test results:', error);
    return {
      success: false,
      message: 'An unexpected error occurred',
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

/**
 * Gets email from URL parameters or returns null
 * @returns Email string or null
 */
export function getEmailFromUrl(): string | null {
  try {
    const params = new URLSearchParams(window.location.search);
    const email = params.get('email');
    return email ? email.trim() : null;
  } catch (error) {
    console.error('Error getting email from URL:', error);
    return null;
  }
}

