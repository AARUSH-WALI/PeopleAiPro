import React, { useEffect, useState } from 'react';
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface RetentionAnalysisProps {
  retentionData: {
    retention_score?: number | null;
    retention_risk?: string | null;
    retention_analysis?: any;
  } | null;
  retentionInsight?: string | null;
}

const getRiskColor = (risk: string | null | undefined) => {
  if (!risk) return 'bg-gray-100 text-gray-800';
  switch (risk.toLowerCase()) {
    case 'low':
      return 'bg-green-100 text-green-800';
    case 'medium':
      return 'bg-amber-100 text-amber-800';
    case 'high':
      return 'bg-red-100 text-red-800';
    default:
      return 'bg-gray-100 text-gray-800';
  }
};

// Retention Score Gauge Component
function RetentionScoreGauge({ score }: { score: number }) {
  const [currentRotation, setCurrentRotation] = useState(0);
  const targetRotation = (score / 100) * 180; // 180 degrees for half circle
  
  // Determine color based on score (higher is better for retention)
  const getColor = () => {
    if (score >= 70) return '#4ade80'; // green - Low risk
    if (score >= 50) return '#facc15'; // yellow - Medium risk
    return '#ef4444'; // red - High risk
  };

  // Animate the gauge needle
  useEffect(() => {
    const animationDuration = 1500;
    const startTime = performance.now();
    const startRotation = currentRotation;
    
    const animate = (currentTime: number) => {
      const elapsedTime = currentTime - startTime;
      const progress = Math.min(elapsedTime / animationDuration, 1);
      
      const easeOutQuad = (t: number) => 1 - (1 - t) * (1 - t);
      const easedProgress = easeOutQuad(progress);
      
      const newRotation = startRotation + (targetRotation - startRotation) * easedProgress;
      setCurrentRotation(newRotation);
      
      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };
    
    requestAnimationFrame(animate);
  }, [score, targetRotation]);

  return (
    <div className="relative w-full mx-auto">
      <svg viewBox="0 0 200 140" className="w-full drop-shadow-md">
        {/* Gauge background */}
        <path
          d="M 10 110 A 90 90 0 0 1 190 110"
          fill="none"
          stroke="#e5e7eb"
          strokeWidth="20"
          strokeLinecap="round"
        />
        
        {/* Red section (0-50%) - High Risk */}
        <path
          d="M 10 110 A 90 90 0 0 1 55 29"
          fill="none"
          stroke="#ef4444"
          strokeWidth="20"
          strokeLinecap="round"
        />
        {/* Yellow section (50-70%) - Medium Risk */}
        <path
          d="M 55 29 A 90 90 0 0 1 100 20"
          fill="none"
          stroke="#facc15"
          strokeWidth="20"
          strokeLinecap="round"
        />
        {/* Green section (70-100%) - Low Risk */}
        <path
          d="M 100 20 A 90 90 0 0 1 190 110"
          fill="none"
          stroke="#4ade80"
          strokeWidth="20"
          strokeLinecap="round"
        />
        
        {/* Score indicators */}
        <text x="10" y="130" textAnchor="middle" fontSize="12" fontWeight="medium" fill="#666">0</text>
        <text x="55" y="25" textAnchor="middle" fontSize="12" fontWeight="medium" fill="#666">50</text>
        <text x="100" y="15" textAnchor="middle" fontSize="12" fontWeight="medium" fill="#666">70</text>
        <text x="190" y="130" textAnchor="middle" fontSize="12" fontWeight="medium" fill="#666">100</text>
        
        {/* Gauge needle pivot point */}
        <circle cx="100" cy="110" r="8" fill="#374151" />
        
        {/* Gauge needle */}
        <g transform={`rotate(${currentRotation - 90} 100 110)`}>
          <line
            x1="100"
            y1="110"
            x2="100"
            y2="30"
            stroke="#1f2937"
            strokeWidth="4"
            strokeLinecap="round"
          />
          <circle cx="100" cy="30" r="6" fill={getColor()} stroke="#1f2937" strokeWidth="1.5" />
        </g>
        
        {/* Score text */}
        <text
          x="100"
          y="85"
          textAnchor="middle"
          fontSize="28"
          fontWeight="bold"
          fill="#374151"
        >
          {score.toFixed(0)}
        </text>
      </svg>
      
      {/* Gauge labels */}
      <div className="flex justify-between mt-2 px-4 text-sm text-gray-600 font-medium">
        <span>High Risk</span>
        <span>Medium</span>
        <span>Low Risk</span>
      </div>
    </div>
  );
}

export default function RetentionAnalysis({ retentionData, retentionInsight }: RetentionAnalysisProps) {
  // Extract data with safe defaults
  const retentionScore = retentionData?.retention_score ?? null;
  const retentionRisk = retentionData?.retention_risk ?? null;
  
  // Parse retention_analysis if it's a string
  let analysisData: any = null;
  if (retentionData?.retention_analysis) {
    if (typeof retentionData.retention_analysis === 'string') {
      try {
        analysisData = JSON.parse(retentionData.retention_analysis);
      } catch (e) {
        console.error('Error parsing retention_analysis:', e);
        analysisData = null;
      }
    } else {
      analysisData = retentionData.retention_analysis;
    }
  }

  // If no data at all, show message
  if (!retentionScore && !retentionRisk && !analysisData) {
    return (
      <Card className="bg-white border-gray-200">
        <CardHeader>
          <CardTitle className="text-2xl font-semibold text-center">Retention Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-center text-gray-500 py-4">Retention analysis not available</p>
          <p className="text-center text-xs text-gray-400 mt-2">Complete the personality test to generate retention analysis</p>
        </CardContent>
      </Card>
    );
  }

  const scoreValue = retentionScore !== null && retentionScore !== undefined ? Number(retentionScore) : 0;
  const riskValue = retentionRisk || 'Unknown';

  return (
    <Card className="bg-white border-gray-200">
      <CardHeader>
        <CardTitle className="text-2xl font-semibold text-center mb-4">Retention Analysis</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Retention Score Gauge */}
        {retentionScore !== null && retentionScore !== undefined && (
          <div className="w-full max-w-[220px] mx-auto">
            <RetentionScoreGauge score={scoreValue} />
          </div>
        )}

        {/* Retention Risk Badge */}
        {retentionRisk && (
          <div className="flex justify-center mt-4">
            <span className={`px-4 py-2 text-sm font-semibold rounded-full ${getRiskColor(retentionRisk)}`}>
              Risk Level: {riskValue}
            </span>
          </div>
        )}

        {/* Detailed Analysis Section */}
        {analysisData && (
          <div className="mt-4">
            <h4 className="text-base font-semibold mb-2 text-gray-800">Detailed Analysis</h4>
            <ScrollArea className="h-[250px] border border-gray-200 rounded-md p-4 bg-gray-50">
              <div className="space-y-3 text-sm">
                {/* Component Scores */}
                {analysisData.component_scores && (
                  <div>
                    <h5 className="font-semibold text-gray-700 mb-2">Component Scores:</h5>
                    <div className="space-y-1 pl-2">
                      {analysisData.component_scores.stability !== undefined && (
                        <div className="text-gray-600">• Stability: {Number(analysisData.component_scores.stability).toFixed(1)}</div>
                      )}
                      {analysisData.component_scores.personality !== undefined && (
                        <div className="text-gray-600">• Personality: {Number(analysisData.component_scores.personality).toFixed(1)}</div>
                      )}
                      {analysisData.component_scores.engagement !== undefined && (
                        <div className="text-gray-600">• Engagement: {Number(analysisData.component_scores.engagement).toFixed(1)}</div>
                      )}
                      {analysisData.component_scores.fitment_factor !== undefined && (
                        <div className="text-gray-600">• Fitment Factor: {Number(analysisData.component_scores.fitment_factor).toFixed(1)}</div>
                      )}
                      {analysisData.component_scores.tier_score !== undefined && (
                        <div className="text-gray-600">• Academic Tier: {Number(analysisData.component_scores.tier_score).toFixed(1)}</div>
                      )}
                      {analysisData.component_scores.peer_comparison !== undefined && analysisData.component_scores.peer_comparison !== null && (
                        <div className="text-gray-600">• Peer Comparison: {Number(analysisData.component_scores.peer_comparison).toFixed(1)}</div>
                      )}
                      {analysisData.peer_group && (
                        <div className="text-gray-500 text-xs mt-1">Peer Group: {analysisData.peer_group}</div>
                      )}
                    </div>
                  </div>
                )}

                {/* Risk Flags */}
                {analysisData.risk_flags && Array.isArray(analysisData.risk_flags) && analysisData.risk_flags.length > 0 && (
                  <div>
                    <h5 className="font-semibold text-gray-700 mb-2">Risk Flags:</h5>
                    <ul className="list-disc pl-5 space-y-1">
                      {analysisData.risk_flags.map((flag: string, index: number) => (
                        <li key={index} className="text-gray-600">{flag}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Insights */}
                {analysisData.insights && Array.isArray(analysisData.insights) && analysisData.insights.length > 0 && (() => {
                  const positiveLines: string[] = [];
                  const negativeLines: string[] = [];

                  const isNegative = (text: string) => {
                    const lower = text.toLowerCase();
                    const negativeKeywords = [
                      'risk',
                      'concern',
                      'issue',
                      'flag',
                      'caution',
                      'monitor',
                      'avoid',
                      'weak',
                      'low ',
                      'marginal',
                      'limited',
                      'gap',
                      'deficit',
                      'low stability',
                      'short tenure',
                      'hopper',
                      'instability',
                      'turnover'
                    ];
                    return negativeKeywords.some(keyword => lower.includes(keyword));
                  };

                  analysisData.insights.forEach((raw: string) => {
                    const normalized = (raw || '').toString().trim();
                    if (!normalized) return;
                    if (isNegative(normalized)) {
                      negativeLines.push(normalized);
                    } else {
                      positiveLines.push(normalized);
                    }
                  });

                  if (positiveLines.length === 0 && negativeLines.length === 0) {
                    return null;
                  }

                  return (
                    <div>
                      <h5 className="font-semibold text-gray-700 mb-2">Likelihood to Stay Analysis</h5>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {positiveLines.length > 0 && (
                          <div className="border border-green-200 bg-green-50 rounded-md p-3">
                            <h6 className="text-sm font-semibold text-green-800 mb-1">Positive indicators</h6>
                            <ul className="list-disc pl-5 space-y-1">
                              {positiveLines.map((line, index) => (
                                <li key={index} className="text-sm text-green-900">
                                  {line}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {negativeLines.length > 0 && (
                          <div className="border border-red-200 bg-red-50 rounded-md p-3">
                            <h6 className="text-sm font-semibold text-red-800 mb-1">Risks & concerns</h6>
                            <ul className="list-disc pl-5 space-y-1">
                              {negativeLines.map((line, index) => (
                                <li key={index} className="text-sm text-red-900">
                                  {line}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}

                {retentionInsight && (() => {
                  const lines = retentionInsight
                    .split('\n')
                    .map(line => line.trim())
                    .filter(line => line.length > 0);

                  const positiveLines: string[] = [];
                  const negativeLines: string[] = [];

                  const isNegative = (text: string) => {
                    const lower = text.toLowerCase();
                    const negativeKeywords = [
                      'risk',
                      'concern',
                      'issue',
                      'flag',
                      'caution',
                      'monitor',
                      'avoid',
                      'weak',
                      'low stability',
                      'short tenure',
                      'hopper',
                      'instability',
                      'turnover'
                    ];
                    return negativeKeywords.some(keyword => lower.includes(keyword));
                  };

                  lines.forEach(rawLine => {
                    const normalized = rawLine.replace(/^[•*-]\s*/, '• ').trim();
                    if (isNegative(normalized)) {
                      negativeLines.push(normalized);
                    } else {
                      positiveLines.push(normalized);
                    }
                  });

                  if (positiveLines.length === 0 && negativeLines.length === 0) {
                    return null;
                  }

                  return (
                    <div>
                      <h5 className="font-semibold text-gray-700 mb-2">Likelihood to Stay Analysis</h5>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {positiveLines.length > 0 && (
                          <div className="border border-green-200 bg-green-50 rounded-md p-3">
                            <h6 className="text-sm font-semibold text-green-800 mb-1">Positive indicators</h6>
                            <ul className="list-disc pl-5 space-y-1">
                              {positiveLines.map((line, index) => (
                                <li key={index} className="text-sm text-green-900">
                                  {line}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {negativeLines.length > 0 && (
                          <div className="border border-red-200 bg-red-50 rounded-md p-3">
                            <h6 className="text-sm font-semibold text-red-800 mb-1">Risks & concerns</h6>
                            <ul className="list-disc pl-5 space-y-1">
                              {negativeLines.map((line, index) => (
                                <li key={index} className="text-sm text-red-900">
                                  {line}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })()}

                {/* Risk Description */}
                {analysisData.risk_description && (
                  <div>
                    <h5 className="font-semibold text-gray-700 mb-2">Risk Assessment:</h5>
                    <p className="text-gray-600 pl-2">{analysisData.risk_description}</p>
                  </div>
                )}

                {/* Tier Details */}
                {analysisData.tier_details && (
                  <div>
                    <h5 className="font-semibold text-gray-700 mb-2">Institution Details:</h5>
                    <div className="space-y-1 pl-2 text-gray-600">
                      {JSON.stringify(analysisData.tier_details, null, 2)}
                    </div>
                  </div>
                )}

                {/* If no structured data, show raw JSON */}
                {!analysisData.component_scores && 
                 !analysisData.risk_flags && 
                 !analysisData.insights && 
                 !analysisData.risk_description && (
                  <div className="text-gray-600">
                    <pre className="text-xs whitespace-pre-wrap">
                      {JSON.stringify(analysisData, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

