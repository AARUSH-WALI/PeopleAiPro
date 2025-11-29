
import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { usePersonalityTest, PersonalityScores } from "@/hooks/usePersonalityTest";

export default function PersonalityTest() {
  const [searchParams] = useSearchParams();
  const token = searchParams.get('token');
  const { isSubmitting, submitTestResults } = usePersonalityTest();
  
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [scores, setScores] = useState<PersonalityScores>({
    extroversion: 50,
    agreeableness: 50,
    openness: 50,
    neuroticism: 50,
    conscientiousness: 50
  });
  const [submitted, setSubmitted] = useState(false);

  const handleScoreChange = (trait: keyof PersonalityScores, value: number[]) => {
    setScores(prev => ({
      ...prev,
      [trait]: value[0]
    }));
  };

  const calculateFitmentScore = () => {
    // Simple algorithm to calculate overall fitment (can be customized)
    const { extroversion, agreeableness, openness, neuroticism, conscientiousness } = scores;
    
    // Simplified scoring formula - can be replaced with your actual algorithm
    // Note: Higher neuroticism often correlates with lower job performance, so we invert it
    const invertedNeuroticism = 100 - neuroticism;
    
    return (extroversion + agreeableness + openness + invertedNeuroticism + conscientiousness) / 5;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!token) {
      alert("Invalid test link. Missing token.");
      return;
    }
    
    if (!name || !email) {
      alert("Please enter your name and email.");
      return;
    }
    
    const fitmentScore = calculateFitmentScore();
    
    const result = await submitTestResults({
      token,
      name,
      email,
      fitmentScore,
      personalityScores: scores
    });
    
    if (result) {
      setSubmitted(true);
    }
  };

  if (!token) {
    return (
      <Card className="max-w-3xl mx-auto mt-8">
        <CardHeader>
          <CardTitle>Invalid Test Link</CardTitle>
          <CardDescription>This personality test link appears to be invalid or expired.</CardDescription>
        </CardHeader>
      </Card>
    );
  }
  
  if (submitted) {
    return (
      <Card className="max-w-3xl mx-auto mt-8">
        <CardHeader>
          <CardTitle>Thank You!</CardTitle>
          <CardDescription>Your personality test has been submitted successfully.</CardDescription>
        </CardHeader>
        <CardContent>
          <p>Your responses have been recorded. We'll be in touch about next steps in the hiring process.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="max-w-3xl mx-auto mt-8">
      <CardHeader>
        <CardTitle>Personality Assessment</CardTitle>
        <CardDescription>Please complete this assessment as part of your application process.</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="name">Full Name</Label>
                <Input 
                  id="name" 
                  value={name} 
                  onChange={(e) => setName(e.target.value)} 
                  required 
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="email">Email Address</Label>
                <Input 
                  id="email" 
                  type="email" 
                  value={email} 
                  onChange={(e) => setEmail(e.target.value)}
                  required 
                />
              </div>
            </div>

            <div className="pt-6">
              <h3 className="text-lg font-medium mb-4">Rate yourself on the following traits:</h3>
              
              <div className="space-y-8">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label>Extroversion</Label>
                    <span className="text-sm">{scores.extraversion}%</span>
                  </div>
                  <Slider 
                    value={[scores.extraversion]} 
                    onValueChange={(value) => handleScoreChange('extroversion', value)} 
                    max={100} 
                    step={1}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Introverted</span>
                    <span>Extroverted</span>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label>Agreeableness</Label>
                    <span className="text-sm">{scores.agreeableness}%</span>
                  </div>
                  <Slider 
                    value={[scores.agreeableness]} 
                    onValueChange={(value) => handleScoreChange('agreeableness', value)} 
                    max={100} 
                    step={1}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Challenging</span>
                    <span>Cooperative</span>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label>Openness</Label>
                    <span className="text-sm">{scores.openness}%</span>
                  </div>
                  <Slider 
                    value={[scores.openness]} 
                    onValueChange={(value) => handleScoreChange('openness', value)} 
                    max={100} 
                    step={1}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Conventional</span>
                    <span>Inventive</span>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label>Neuroticism</Label>
                    <span className="text-sm">{scores.neuroticism}%</span>
                  </div>
                  <Slider 
                    value={[scores.neuroticism]} 
                    onValueChange={(value) => handleScoreChange('neuroticism', value)} 
                    max={100} 
                    step={1}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Confident</span>
                    <span>Nervous</span>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label>Conscientiousness</Label>
                    <span className="text-sm">{scores.conscientiousness}%</span>
                  </div>
                  <Slider 
                    value={[scores.conscientiousness]} 
                    onValueChange={(value) => handleScoreChange('conscientiousness', value)} 
                    max={100} 
                    step={1}
                  />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Spontaneous</span>
                    <span>Organized</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </form>
      </CardContent>
      <CardFooter>
        <Button 
          type="submit" 
          onClick={handleSubmit} 
          disabled={isSubmitting || !name || !email}
          className="w-full"
        >
          {isSubmitting ? "Submitting..." : "Submit Assessment"}
        </Button>
      </CardFooter>
    </Card>
  );
}
