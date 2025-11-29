// Service to determine institution tiers using Gemini API
// Only uses tiers 1, 2, and 3 (NULL if empty/not found)

/**
 * Determine institution tier using Gemini API
 * Returns 1, 2, 3, or null
 * - Tier 1: Top-tier institutions (IITs, IIMs, top universities)
 * - Tier 2: Good institutions (NITs, state universities, good private colleges)
 * - Tier 3: Other institutions (anything below tier 2, or unknown)
 * - null: If institution name is empty or not provided
 */
export async function determineInstitutionTier(institutionName, apiKey) {
    // Return null if institution name is empty or not provided
    if (!institutionName || !institutionName.trim()) {
        return null;
    }

    // If no API key provided, default to tier 3
    if (!apiKey) {
        console.warn('Gemini API key not provided, defaulting to tier 3');
        return 3;
    }

    try {
        const geminiEndpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;

        const prompt = `
      Determine the tier of the following educational institution in India.
      
      Institution Name: ${institutionName}
      
      Classify it into one of these tiers ONLY:
      - Tier 1: Top-tier institutions (IITs, IIMs, AIIMS, IISc, top central/state universities, premier engineering/medical colleges)
      - Tier 2: Good institutions (NITs, good state universities, reputable private universities, established engineering/medical colleges)
      - Tier 3: Other institutions (local colleges, lesser-known institutions, or if you're uncertain about the institution's reputation)
      
      IMPORTANT RULES:
      1. Return ONLY the tier number (1, 2, or 3) - no text, no explanation
      2. If the institution name is unclear or you're not certain, return 3
      3. If you cannot determine the tier confidently, return 3
      4. Only return 1, 2, or 3 - nothing else
      
      Return format: Just the number (1, 2, or 3)
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
                    temperature: 0.1,
                    maxOutputTokens: 10,
                    topP: 0.95,
                    topK: 40
                }
            })
        });

        if (!response.ok) {
            throw new Error(`Gemini API error: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error.message || 'Failed to determine tier');
        }

        // Extract the tier number from response
        const responseText = data.candidates[0]?.content?.parts[0]?.text?.trim() || '';

        // Extract just the number (1, 2, or 3)
        const tierMatch = responseText.match(/\b([123])\b/);

        if (tierMatch) {
            const tier = parseInt(tierMatch[1], 10);
            // Ensure it's 1, 2, or 3
            if (tier >= 1 && tier <= 3) {
                return tier;
            }
        }

        // Default to tier 3 if parsing fails
        console.warn(`Could not parse tier from response: ${responseText}, defaulting to tier 3`);
        return 3;
    } catch (error) {
        console.error('Error determining institution tier:', error);
        // On error, default to tier 3
        return 3;
    }
}

/**
 * Determine tiers for both UG and PG institutions
 * Returns { ug_tier: number|null, pg_tier: number|null }
 */
export async function determineInstitutionTiers(ugInstitute, pgInstitute, apiKey) {
    const [ugTier, pgTier] = await Promise.all([
        determineInstitutionTier(ugInstitute, apiKey),
        determineInstitutionTier(pgInstitute, apiKey)
    ]);

    return {
        ug_tier: ugTier,
        pg_tier: pgTier
    };
}



