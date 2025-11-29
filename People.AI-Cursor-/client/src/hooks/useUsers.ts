
import { useState, useEffect } from "react";
import { useToast } from "@/components/ui/use-toast";

export interface UserProfile {
  id: string;
  name: string;
  email: string;
  score: number;
  jobRole?: string;
  experience?: string;
  education?: string;
  about?: string;
  profileImage?: string;
  personalityScores?: {
    extraversion: number;
    agreeableness: number;
    openness: number;
    neuroticism: number;
    conscientiousness: number;
  };
}

export function useUsers() {
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const { toast } = useToast();

  const fetchUsers = async () => {
    setIsLoading(true);
    try {
      // Using mock data - replace with your actual API calls
      const mockUsers: UserProfile[] = [
        {
          id: "1",
          name: "Saksham Gupta",
          email: "2022a6041@mietjammu.in",
          score: 75.89,
          jobRole: "Assistant Professor",
          experience: "3+ Years",
          education: "Ph.D. in Computer Science",
          about: "Experienced educator with a focus on AI and machine learning",
          personalityScores: {
            extraversion: 76,
            agreeableness: 84,
            openness: 69,
            neuroticism: 50,
            conscientiousness: 90
          }
        },
        {
          id: "2",
          name: "Ayush Thakur",
          email: "ayushthakur1412@gmail.com",
          score: 69.94,
          jobRole: "Professor",
          experience: "6+ Years",
          education: "Master of Computer Science",
          about: "Self-experienced Front End Developer with a strong background in software and web development",
          personalityScores: {
            extraversion: 68,
            agreeableness: 72,
            openness: 85,
            neuroticism: 42,
            conscientiousness: 81
          }
        },
        {
          id: "3",
          name: "Dhruv Gupta",
          email: "2022a6015@mietjammu.in",
          score: 59.99,
          jobRole: "Software Engineer",
          experience: "4+ Years",
          education: "M.Tech in Software Engineering",
          about: "Software engineer with expertise in full-stack development.",
          personalityScores: {
            extraversion: 45,
            agreeableness: 60,
            openness: 55,
            neuroticism: 25,
            conscientiousness: 70
          }
        }
      ];

      setUsers(mockUsers);
    } catch (error) {
      toast({
        title: "Error fetching users",
        description: "Please try again later.",
        variant: "destructive",
      });
      console.error('Error fetching users:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  return { users, isLoading, fetchUsers };
}
