// Service to generate AI-powered retention insights using Gemini API
// Returns comprehensive bullet-point insights covering positives, negatives, and risk flags for HR professionals

/**
 * Generate comprehensive retention insights using Gemini AI
 * @param {Object} retentionData - Retention analysis data
 * @param {number} retentionData.retention_score - Retention score (0-100)
 * @param {string} retentionData.retention_risk - Risk level (Low/Medium/High)
 * @param {Object} retentionData.component_scores - Component scores breakdown
 * @param {Array} retentionData.risk_flags - Array of risk flags
 * @param {string} apiKey - Gemini API key
 * @returns {Promise<string>} - Detailed bullet-point insights
 */
export async function generateRetentionInsight(retentionData, apiKey) {
    if (!apiKey) {
        console.warn('Gemini API key not provided, returning default insight');
        return getDefaultInsight(retentionData.retention_risk);
    }

    try {
        const geminiEndpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;

        // Add timestamp to prompt to ensure fresh analysis each time
        const timestamp = new Date().toISOString();
        
        const prompt = `
You are an HR hiring analyst. Generate extremely short, crisp insights (6–12 words each) that help HR decide whether to hire the candidate.

STRICT RULES:
- ABSOLUTELY NO long sentences.
- Each bullet MUST be 6–12 words ONLY.
- No soft, vague language like “suggests,” “indicates,” or “alignment potential.”
- Use direct HR decision language: “Good fit,” “Risky hire,” “Strong stability,” etc.
- No explanations. No filler. No repeated ideas.
- Do NOT exceed 6 bullet points.

CURRENT ANALYSIS TIME: ${timestamp}

RETENTION DATA:
- Retention Score: ${retentionData.retention_score}/100
- Risk Level: ${retentionData.retention_risk}
- Component Scores:
  * Job Stability: ${retentionData.component_scores?.stability || 'N/A'}/100
  * Personality Fit: ${retentionData.component_scores?.personality || 'N/A'}/100
  * Professional Engagement: ${retentionData.component_scores?.engagement || 'N/A'}/100
  * Fitment Factor: ${retentionData.component_scores?.fitment_factor || 'N/A'}/100
${retentionData.risk_flags && retentionData.risk_flags.length > 0 ? `- Risk Flags: ${retentionData.risk_flags.join(', ')}` : ''}

OUTPUT FORMAT (MANDATORY):
• Strength (max 12 words)
• Strength (max 12 words)
• Risk (max 12 words)
• Risk (max 12 words)
• Risk flag insight (max 12 words)
• Final HR recommendation (max 12 words)

Your bullets MUST be short, punchy, and hiring-focused.
Return ONLY the bullet points, nothing else.


// You are an expert HR retention analyst. Analyze the candidate's retention data and generate clear, meaningful insights for HR in short bullet points.

// CURRENT ANALYSIS TIME: ${timestamp}

// RETENTION DATA:
// - Retention Score: ${retentionData.retention_score}/100
// - Risk Level: ${retentionData.retention_risk}
// - Component Scores:
//   * Job Stability: ${retentionData.component_scores?.stability || 'N/A'}/100
//   * Personality Fit: ${retentionData.component_scores?.personality || 'N/A'}/100
//   * Professional Engagement: ${retentionData.component_scores?.engagement || 'N/A'}/100
//   * Fitment Factor: ${retentionData.component_scores?.fitment_factor || 'N/A'}/100
// ${retentionData.risk_flags && retentionData.risk_flags.length > 0 ? `- Risk Flags: ${retentionData.risk_flags.join(', ')}` : ''}

// INSTRUCTIONS:
// 1. Provide insights only in bullet points.
// 2. Keep every bullet short — **max 12–16 words**.
// 3. Ensure bullets are still meaningful and actionable.
// 4. Include both strengths and concerns.
// 5. Address every risk flag directly.
// 6. Avoid long explanations; use tight, punchy HR language.
// 7. Include 5–7 bullets total.

// OUTPUT FORMAT:
// • Positive aspect  
// • Positive aspect  
// • Concern  
// • Concern  
// • Risk flag insight  
// • Overall recommendation  

// CRITICAL RULES:
// - No headers, no extra text — only bullets starting with "•".
// - No long sentences; split ideas if needed.
// - Stick strictly to retention-related insights.


// You are an expert HR retention analyst. Analyze the candidate's retention data and provide comprehensive insights in bullet points that HR can understand clearly.

// CURRENT ANALYSIS TIME: ${timestamp}

// RETENTION DATA:
// - Retention Score: ${retentionData.retention_score}/100
// - Risk Level: ${retentionData.retention_risk}
// - Component Scores:
//   * Job Stability: ${retentionData.component_scores?.stability || 'N/A'}/100
//   * Personality Fit: ${retentionData.component_scores?.personality || 'N/A'}/100
//   * Professional Engagement: ${retentionData.component_scores?.engagement || 'N/A'}/100
//   * Fitment Factor: ${retentionData.component_scores?.fitment_factor || 'N/A'}/100
// ${retentionData.risk_flags && retentionData.risk_flags.length > 0 ? `- Risk Flags: ${retentionData.risk_flags.join(', ')}` : ''}

// INSTRUCTIONS:
// 1. Analyze the retention score, risk level, component scores, and risk flags
// 2. Provide insights in bullet point format (use • or - for bullets)
// 3. Include BOTH positive aspects (strengths) and negative aspects (concerns)
// 4. Address each risk flag specifically if present
// 5. Be clear and actionable for HR professionals
// 6. Keep each bullet point concise (1-2 sentences max)
// 7. Focus on retention-related insights

// OUTPUT FORMAT:
// Provide insights in this structure:
// • [Positive aspect 1]
// • [Positive aspect 2]
// • [Concern/Issue 1]
// • [Concern/Issue 2]
// • [Risk flag insight if applicable]
// • [Overall recommendation]

// EXAMPLES OF GOOD INSIGHTS:
// For High Risk Candidate:
// • Low job stability with frequent position changes detected
// • Below average conscientiousness score indicates potential commitment issues
// • Limited professional development activities observed
// • Recommend structured onboarding and close monitoring
// • Consider mentorship program to improve engagement

// For Low Risk Candidate:
// • Strong job stability with consistent tenure history
// • High conscientiousness and agreeableness scores indicate reliable performer
// • Active professional development with workshops and trainings
// • Good personality fit for team collaboration
// • Low retention risk - suitable for long-term projects

// CRITICAL RULES:
// - Return ONLY bullet points (no headers, no explanations)
// - Include 4-6 bullet points total
// - Mix of positive and negative aspects
// - Address risk flags if present
// - Use professional HR language
// - Be specific to the data provided
// - Each bullet should be clear and actionable

Return format: Just the bullet points, one per line, starting with • or -
`;

        const response = await fetch(geminiEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                contents: [
                    {
                        parts: [{ text: prompt }]
                    }
                ],
                generationConfig: {
                    temperature: 0.5, // Balanced temperature for detailed but focused output
                    maxOutputTokens: 300, // Allow for detailed bullet points
                    topP: 0.9,
                    topK: 40
                }
            })
        });

        if (!response.ok) {
            throw new Error(`Gemini API error: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error.message || 'Gemini API returned an error');
        }

        const text = data.candidates?.[0]?.content?.parts?.[0]?.text || '';
        
        console.log(`[retentionInsightsService] Raw Gemini response: "${text}"`);
        
        if (!text) {
            throw new Error('No text in Gemini response');
        }

        // Clean and format the insights (bullet points)
        let cleanedText = text.trim()
            .replace(/^["']|["']$/g, '') // Remove quotes
            .replace(/^```[\w]*\n?/gm, '') // Remove code block markers
            .replace(/```$/gm, '') // Remove trailing code block markers
            .replace(/^Insights?:\s*/i, '') // Remove "Insight:" prefix if present
            .replace(/^Analysis:\s*/i, '') // Remove "Analysis:" prefix if present
            .trim();

        // Ensure bullet points are properly formatted
        // Convert various bullet formats to consistent format
        cleanedText = cleanedText
            .replace(/^[-*]\s+/gm, '• ') // Convert - or * to •
            .replace(/^(\d+\.)\s+/gm, '• ') // Convert numbered lists to bullets
            .split('\n')
            .filter(line => line.trim().length > 0) // Remove empty lines
            .map(line => {
                // Ensure line starts with bullet
                const trimmed = line.trim();
                if (!trimmed.startsWith('•') && !trimmed.startsWith('-') && !trimmed.startsWith('*')) {
                    return '• ' + trimmed;
                }
                return trimmed;
            })
            .join('\n');

        // Limit to 6 bullet points max
        const lines = cleanedText.split('\n').filter(line => line.trim().length > 0);
        const finalInsight = lines.slice(0, 6).join('\n') || getDefaultInsight(retentionData.retention_risk);
        
        console.log(`[retentionInsightsService] Final cleaned insight:\n${finalInsight}`);
        
        return finalInsight;

    } catch (error) {
        console.error('Error generating retention insight:', error);
        return getDefaultInsight(retentionData.retention_risk);
    }
}

/**
 * Get default insight based on risk level when Gemini API fails
 */
function getDefaultInsight(riskLevel) {
    const risk = riskLevel?.toLowerCase() || 'medium';
    
    if (risk === 'low') {
        return '• Strong retention indicators observed\n• Good job stability and personality fit\n• Low risk candidate - suitable for long-term roles\n• Recommend standard onboarding process';
    } else if (risk === 'high') {
        return '• High retention risk detected\n• Requires close monitoring and support\n• Recommend structured onboarding program\n• Consider mentorship and regular check-ins';
    } else {
        return '• Moderate retention risk identified\n• Some areas need attention\n• Recommend active monitoring\n• Provide development opportunities';
    }
}


