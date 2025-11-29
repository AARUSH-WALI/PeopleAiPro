// Enhanced Retention Risk Scorer with Peer Comparison
// Converted from Python for Node.js backend integration
// Supports peer group comparison using historical data

/**
 * Calculate retention risk based on multiple factors
 * Uses research-backed indicators with peer comparison functionality
 */
class RetentionScorer {
    constructor() {
        this.riskCategories = {
            Low: 'Low Risk - High retention likelihood',
            Medium: 'Medium Risk - Monitor and support',
            High: 'High Risk - Intervention recommended'
        };
        this.peerGroups = {};
        this.candidateAssessments = {};
    }

    /**
     * Load peer groups from JSON file (converted from .pkl)
     * @param {Object} peerGroupsData - Peer groups data loaded from JSON
     */
    loadPeerGroups(peerGroupsData) {
        if (peerGroupsData && typeof peerGroupsData === 'object') {
            this.peerGroups = peerGroupsData;
            console.log(`✓ Loaded ${Object.keys(this.peerGroups).length} peer groups`);
        }
    }

    /**
     * Add a peer group for comparison
     */
    addPeerGroup(groupName, candidatesData) {
        const validatedData = candidatesData.filter(c => this._validateCandidateData(c));
        if (validatedData.length === 0) {
            throw new Error(`No valid candidates in peer group '${groupName}'`);
        }
        this.peerGroups[groupName] = validatedData;
    }

    /**
     * Validate required fields exist
     */
    _validateCandidateData(candidateData) {
        const requiredFields = [
            'longevity_years', 'number_of_unique_designations',
            'workshops', 'trainings', 'fitment_score'
        ];
        return requiredFields.every(field => candidateData.hasOwnProperty(field));
    }

    /**
     * Normalize inconsistent tier key naming
     */
    _normalizeTierKeys(candidateData) {
        const data = { ...candidateData };
        if ('TierUG' in data && !('UG_Tier' in data)) {
            data.UG_Tier = data.TierUG;
            delete data.TierUG;
        }
        if ('TierPG' in data && !('PG_Tier' in data)) {
            data.PG_Tier = data.TierPG;
            delete data.TierPG;
        }
        return data;
    }

    /**
     * Extract scores for a given type from peer group
     */
    _getPeerScoresForType(groupName, scoreType) {
        if (!this.peerGroups[groupName]) {
            return [];
        }

        const candidates = this.peerGroups[groupName];
        const scores = [];

        for (const candidate of candidates) {
            if (scoreType === 'retention') {
                scores.push(candidate.retention_score || 0);
            } else if (['stability', 'personality', 'engagement', 'fitment_factor', 'tier_score'].includes(scoreType)) {
                const componentScores = candidate.component_scores || {};
                scores.push(componentScores[scoreType] || 0);
            }
        }

        return scores.filter(s => typeof s === 'number' && s >= 0);
    }

    /**
     * Calculate statistics for a peer group
     */
    calculatePeerStatistics(groupName, scoreType) {
        const scores = this._getPeerScoresForType(groupName, scoreType);

        if (scores.length === 0) {
            return {
                mean: 0,
                median: 0,
                std_dev: 0,
                min: 0,
                max: 0,
                count: 0
            };
        }

        const sorted = [...scores].sort((a, b) => a - b);
        const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
        const median = sorted.length % 2 === 0
            ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
            : sorted[Math.floor(sorted.length / 2)];

        const variance = scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length;
        const stdDev = Math.sqrt(variance);

        return {
            mean: Math.round(mean * 100) / 100,
            median: Math.round(median * 100) / 100,
            std_dev: scores.length > 1 ? Math.round(stdDev * 100) / 100 : 0.0,
            min: Math.round(Math.min(...scores) * 100) / 100,
            max: Math.round(Math.max(...scores) * 100) / 100,
            count: scores.length
        };
    }

    /**
     * Calculate percentile rank within peer group
     */
    getPercentileRank(score, groupName, scoreType) {
        const scores = this._getPeerScoresForType(groupName, scoreType);

        if (scores.length === 0) {
            return {
                percentile: null,
                performance: 'Unknown',
                rank: 0,
                total_in_group: 0
            };
        }

        const sortedScores = [...scores].sort((a, b) => a - b);
        const rank = sortedScores.filter(s => s <= score).length;
        const percentile = Math.round((rank / sortedScores.length) * 100 * 100) / 100;

        let performance;
        if (percentile >= 75) {
            performance = 'Top Performer';
        } else if (percentile >= 50) {
            performance = 'Above Average';
        } else if (percentile >= 25) {
            performance = 'Average';
        } else {
            performance = 'Below Average';
        }

        return {
            percentile,
            performance,
            rank,
            total_in_group: sortedScores.length
        };
    }

    /**
     * Calculate job stability indicator (0-100)
     */
    calculateJobStabilityScore(candidateData) {
        const longevity = Math.max(0, candidateData.longevity_years || 0);
        const uniqueJobs = Math.max(1, candidateData.number_of_unique_designations || 1);

        const avgTenure = longevity / uniqueJobs;

        if (avgTenure >= 4) {
            return 100.0;
        } else if (avgTenure >= 2.5) {
            return 80.0;
        } else if (avgTenure >= 1.5) {
            return 60.0;
        } else if (avgTenure >= 1) {
            return 40.0;
        } else {
            return 20.0;
        }
    }

    /**
     * Calculate academic tier indicator (0-100)
     */
    calculateTierScore(candidateData) {
        const tierUG = candidateData.UG_Tier || candidateData.ug_tier || 3;
        const tierPG = candidateData.PG_Tier || candidateData.pg_tier || 3;

        // Normalize tiers to 1-3 only
        const ugTier = (tierUG >= 1 && tierUG <= 3) ? tierUG : 3;
        const pgTier = (tierPG >= 1 && tierPG <= 3) ? tierPG : 3;

        const ugRawScore = { 1: 2.0, 2: 1.0, 3: 0.5 }[ugTier] || 0.5;
        const pgRawScore = { 1: 3.0, 2: 2.0, 3: 1.0 }[pgTier] || 1.0;

        const combinedRawScore = ugRawScore + pgRawScore;
        const tierScore = (combinedRawScore / 5.0) * 100;

        return tierScore;
    }

    /**
     * Calculate retention likelihood from personality (0-100)
     */
    calculatePersonalityRetentionScore(big5Scores) {
        // Normalize scores to 0-40 range (from -50 to 50 range)
        const normalizeScore = (score) => {
            if (score === null || score === undefined || isNaN(Number(score))) return 20;
            const clamped = Math.max(-50, Math.min(50, Number(score)));
            return Math.round(((clamped + 50) / 100) * 40);
        };

        const conscientiousness = Math.max(0, Math.min(40, normalizeScore(big5Scores.conscientiousness)));
        const agreeableness = Math.max(0, Math.min(40, normalizeScore(big5Scores.agreeableness)));
        const neuroticism = Math.max(0, Math.min(40, normalizeScore(big5Scores.neuroticism)));

        const conscientiousnessNorm = (conscientiousness / 40) * 100;
        const agreeablenessNorm = (agreeableness / 40) * 100;
        const neuroticismNorm = ((40 - neuroticism) / 40) * 100;

        const personalityRetention = (
            conscientiousnessNorm * 0.50 +
            agreeablenessNorm * 0.35 +
            neuroticismNorm * 0.15
        );

        return personalityRetention;
    }

    /**
     * Calculate professional engagement indicator (0-100)
     */
    calculateEngagementScore(candidateData) {
        const workshops = Math.max(0, candidateData.workshops || candidateData.workshops_count || 0);
        const trainings = Math.max(0, candidateData.trainings || candidateData.trainings_count || 0);
        const papers = Math.max(0, candidateData.total_papers || candidateData.research_papers_count || 0);
        const patents = Math.max(0, candidateData.total_patents || candidateData.patents_count || 0);
        const achievements = Math.max(0, candidateData.achievements || candidateData.achievements_count || 0);
        const experience = Math.max(1, candidateData.average_experience || candidateData.longevity_years || 1);

        const workshopScore = experience > 0 ? Math.min(((workshops / experience) / 1.0) * 100, 100) : 0;
        const trainingScore = experience > 0 ? Math.min(((trainings / experience) / 1.5) * 100, 100) : 0;
        const papersScore = experience > 0 ? Math.min(((papers / experience) / 1.0) * 100, 100) : 0;
        const patentsScore = experience > 0 ? Math.min(((patents / experience) / 0.5) * 100, 100) : 0;
        const achievementsScore = experience > 0 ? Math.min(((achievements / experience) / 1.0) * 100, 100) : 0;

        const engagement = (
            workshopScore * 0.25 +
            trainingScore * 0.30 +
            papersScore * 0.20 +
            patentsScore * 0.10 +
            achievementsScore * 0.15
        );

        return Math.min(Math.round(engagement * 100) / 100, 100.0);
    }

    /**
     * Adjust retention based on fitment score and category
     */
    calculateFitmentFactor(fitmentScore, category) {
        const normalizedScore = Math.min(Math.max(fitmentScore, 0), 100);

        if (category === 'Experienced') {
            if (normalizedScore >= 85) {
                return 95.0;
            } else if (normalizedScore >= 75) {
                return 82.0;
            } else if (normalizedScore >= 65) {
                return 70.0;
            } else if (normalizedScore >= 55) {
                return 58.0;
            } else if (normalizedScore >= 45) {
                return 45.0;
            } else {
                return 35.0;
            }
        } else {
            if (normalizedScore >= 85) {
                return 92.0;
            } else if (normalizedScore >= 75) {
                return 80.0;
            } else if (normalizedScore >= 65) {
                return 68.0;
            } else if (normalizedScore >= 55) {
                return 56.0;
            } else if (normalizedScore >= 45) {
                return 44.0;
            } else {
                return 32.0;
            }
        }
    }

    /**
     * Identify specific retention risk flags
     */
    identifyRiskFlags(candidateData, big5Scores) {
        const flags = [];

        // Normalize Big5 scores
        const normalizeScore = (score) => {
            if (score === null || score === undefined || isNaN(Number(score))) return 20;
            const clamped = Math.max(-50, Math.min(50, Number(score)));
            return Math.round(((clamped + 50) / 100) * 40);
        };

        const uniqueJobs = candidateData.number_of_unique_designations || candidateData.number_of_jobs || 0;
        const longevity = candidateData.longevity_years || 0;

        if (uniqueJobs >= 4 && (longevity / Math.max(uniqueJobs, 1)) < 1.5) {
            flags.push('Job Hopper Pattern');
        }

        const workshops = candidateData.workshops || candidateData.workshops_count || 0;
        const trainings = candidateData.trainings || candidateData.trainings_count || 0;
        if (workshops + trainings < 2) {
            flags.push('Low Professional Development');
        }

        const conscientiousness = normalizeScore(big5Scores.conscientiousness);
        if (conscientiousness < 20) {
            flags.push('Low Conscientiousness (Retention Risk)');
        }

        const neuroticism = normalizeScore(big5Scores.neuroticism);
        if (neuroticism > 30) {
            flags.push('High Emotional Instability');
        }

        if (candidateData.fitment_score < 45) {
            flags.push('Low Overall Fitment');
        }

        if ((longevity / Math.max(uniqueJobs, 1)) < 1.5) {
            flags.push('Short Average Tenure');
        }

        const ugTier = candidateData.UG_Tier || candidateData.ug_tier || 3;
        const pgTier = candidateData.PG_Tier || candidateData.pg_tier || 3;
        if (ugTier >= 3 || pgTier >= 3) {
            flags.push('Low Academic Tier Institution');
        }

        return flags;
    }

    /**
     * Generate actionable insights based on retention analysis
     */
    generateRetentionInsights(retentionData) {
        const insights = [];
        const riskLevel = retentionData.retention_risk;
        const riskScore = retentionData.retention_score;
        const flags = retentionData.risk_flags;

        if (riskLevel === 'High') {
            insights.push('⚠ HIGH RETENTION RISK - Immediate intervention recommended');
            insights.push('→ Consider structured onboarding and mentorship program');
            insights.push('→ Schedule regular check-ins (bi-weekly for first 6 months)');
        } else if (riskLevel === 'Medium') {
            insights.push('⚡ MEDIUM RETENTION RISK - Active monitoring advised');
            insights.push('→ Provide clear career progression path');
            insights.push('→ Encourage participation in professional development');
        } else {
            insights.push('✅ LOW RETENTION RISK - Strong retention indicators');
            insights.push('→ Leverage for team stability and mentorship roles');
        }

        if (flags.includes('Job Hopper Pattern')) {
            insights.push('→ Address: Frequent job changes - Discuss long-term goals early');
        }
        if (flags.includes('Low Professional Development')) {
            insights.push('→ Address: Limited development activities - Offer training stipend');
        }

        return insights;
    }

    /**
     * Calculate peer comparison score (0-100)
     */
    calculatePeerComparisonScore(retentionScore, groupName) {
        if (!this.peerGroups[groupName]) {
            return null;
        }

        const percentileData = this.getPercentileRank(retentionScore, groupName, 'retention');

        if (percentileData.percentile === null) {
            return null;
        }

        const percentile = percentileData.percentile;

        if (percentile >= 75) {
            return 100.0;
        } else if (percentile >= 50) {
            return 80.0;
        } else if (percentile >= 25) {
            return 60.0;
        } else {
            return 40.0;
        }
    }

    /**
     * Determine appropriate peer group based on candidate data
     */
    determinePeerGroup(candidateData) {
        const category = candidateData.candidate_type || candidateData.category || 'Fresher';
        const longevity = candidateData.longevity_years || 0;

        // Try category-based peer group first
        const categoryGroup = `${category}_Professionals`;
        if (this.peerGroups[categoryGroup]) {
            return categoryGroup;
        }

        // Fall back to experience-based peer groups
        if (longevity <= 3) {
            return 'Junior_Professionals';
        } else if (longevity <= 8) {
            return 'Mid_Level_Professionals';
        } else {
            return 'Senior_Professionals';
        }
    }

    /**
     * Calculate comprehensive retention risk assessment
     */
    calculateRetentionRisk(candidateData, fitmentScore, big5Scores, category, peerGroup = null, candidateId = null) {
        candidateData = this._normalizeTierKeys(candidateData);

        const stabilityScore = this.calculateJobStabilityScore(candidateData);
        const personalityScore = this.calculatePersonalityRetentionScore(big5Scores);
        const engagementScore = this.calculateEngagementScore(candidateData);
        const fitmentFactor = this.calculateFitmentFactor(fitmentScore, category);
        const tierScore = this.calculateTierScore(candidateData);

        // Determine peer group if not provided
        const effectivePeerGroup = peerGroup || this.determinePeerGroup({ ...candidateData, candidate_type: category });

        // Calculate initial retention score for peer comparison
        let initialRetentionScore = (
            stabilityScore * 0.25 +
            personalityScore * 0.35 +
            engagementScore * 0.20 +
            fitmentFactor * 0.15 +
            tierScore * 0.05
        );

        // Calculate peer comparison score if peer group exists
        let peerComparisonScore = null;
        if (effectivePeerGroup && this.peerGroups[effectivePeerGroup]) {
            peerComparisonScore = this.calculatePeerComparisonScore(initialRetentionScore, effectivePeerGroup);
        }

        // Calculate final retention score (with or without peer comparison)
        let retentionScore;
        if (peerComparisonScore !== null) {
            retentionScore = (
                stabilityScore * 0.20 +
                personalityScore * 0.28 +
                engagementScore * 0.16 +
                fitmentFactor * 0.12 +
                tierScore * 0.04 +
                peerComparisonScore * 0.20
            );
        } else {
            retentionScore = initialRetentionScore;
        }

        // Determine risk level
        let riskLevel;
        if (retentionScore >= 70) {
            riskLevel = 'Low';
        } else if (retentionScore >= 50) {
            riskLevel = 'Medium';
        } else {
            riskLevel = 'High';
        }

        // Identify risk flags
        const riskFlags = this.identifyRiskFlags(candidateData, big5Scores);

        // Generate insights
        const insights = this.generateRetentionInsights({
            retention_risk: riskLevel,
            retention_score: retentionScore,
            risk_flags: riskFlags
        });

        // Compile results
        const result = {
            candidate_id: candidateId,
            retention_score: Math.round(retentionScore * 100) / 100,
            retention_risk: riskLevel,
            risk_description: this.riskCategories[riskLevel],
            component_scores: {
                stability: Math.round(stabilityScore * 100) / 100,
                personality: Math.round(personalityScore * 100) / 100,
                engagement: Math.round(engagementScore * 100) / 100,
                fitment_factor: Math.round(fitmentFactor * 100) / 100,
                tier_score: Math.round(tierScore * 100) / 100,
                peer_comparison: peerComparisonScore !== null ? Math.round(peerComparisonScore * 100) / 100 : null
            },
            risk_flags: riskFlags,
            flag_count: riskFlags.length,
            peer_group: effectivePeerGroup,
            insights: insights
        };

        // Store assessment if candidate ID provided
        if (candidateId) {
            this.candidateAssessments[candidateId] = result;
        }

        return result;
    }
}

// Export singleton instance
export const retentionScorer = new RetentionScorer();
export default retentionScorer;
