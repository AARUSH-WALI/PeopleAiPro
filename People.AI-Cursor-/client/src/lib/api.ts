import axios from 'axios';

// Use production API URL in production environment, otherwise use development URL
const API_BASE_URL = import.meta.env.PROD 
  ? import.meta.env.VITE_PROD_API_BASE_URL 
  : import.meta.env.VITE_DEV_API_BASE_URL || 'http://localhost:5000/api';

export interface FitmentCriteria {
  id: string;
  best_fit: number;
  average_fit: number;
  not_fit: number;
  created_at: string;
  updated_at: string;
}

export const criteriaApi = {
  // Get current criteria
  getCriteria: async (): Promise<FitmentCriteria> => {
    const response = await axios.get(`${API_BASE_URL}/criteria`);
    return response.data;
  },

  // Update criteria
  updateCriteria: async (data: {
    best_fit: number;
    average_fit: number;
    not_fit: number;
  }): Promise<FitmentCriteria> => {
    const response = await axios.put(`${API_BASE_URL}/criteria`, data);
    return response.data;
  },
};
