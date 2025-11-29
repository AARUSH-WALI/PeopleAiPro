import { ResumeData } from "@/types/resume";
import axios, { AxiosError } from 'axios';

// Configure axios defaults
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:5000/api',
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
  withCredentials: true,
});

// Add a request interceptor to handle form data
api.interceptors.request.use(
  (config) => {
    // If the data is FormData, remove the Content-Type header
    // to let the browser set it with the correct boundary
    if (config.data instanceof FormData) {
      if (config.headers) {
        delete config.headers['Content-Type'];
      }
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add a response interceptor to handle errors consistently
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError<{ message?: string; error?: string; details?: string }>) => {
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      const errorMessage = error.response.data?.message 
        || error.response.data?.error 
        || error.response.data?.details
        || `Server error: ${error.response.status} ${error.response.statusText}`;
      console.error('API Error Response:', {
        status: error.response.status,
        data: error.response.data,
        message: errorMessage
      });
      return Promise.reject(new Error(errorMessage));
    } else if (error.request) {
      // The request was made but no response was received
      return Promise.reject(new Error('No response from server. Please check your connection.'));
    } else {
      // Something happened in setting up the request that triggered an Error
      return Promise.reject(error);
    }
  }
);

interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
}

export const saveResume = async (
  resumeData: ResumeData, 
  file?: File
): Promise<ApiResponse<{
  id: string;
  file_url?: string;
  file_path?: string;
  original_filename?: string;
  file_size?: number;
  mime_type?: string;
}>> => {
  const formData = new FormData();
  
  try {
    // Add resume data as JSON
    formData.append('data', JSON.stringify(resumeData));
    
    // If there's a file, add it to the form data with field name 'file' to match the server's multer configuration
    if (file) {
      formData.append('file', file);
    }
    
    // Make the request with form data
    // The interceptor will handle the Content-Type header for FormData
    const response = await api.post('/resumes', formData, {
      headers: {
        // The interceptor will handle the Content-Type header
      },
      // Required for Node.js 18+ when sending a body with certain content types
      // @ts-ignore - The type definition might not include this option yet
      duplex: 'half'
    });
    
    return { 
      success: true, 
      data: response.data 
    };
  } catch (error) {
    console.error('Error saving resume:', error);
    const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
    return { 
      success: false, 
      error: errorMessage 
    };
  }
};

export const getResume = async (id: string): Promise<ApiResponse<ResumeData>> => {
  try {
    const response = await api.get(`/resumes/${id}`);
    return { 
      success: true, 
      data: response.data 
    };
  } catch (error) {
    console.error('Error fetching resume:', error);
    const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
    return { 
      success: false, 
      error: errorMessage 
    };
  }
};

interface SearchResult {
  id: string;
  name: string;
  email: string;
  best_fit_for: string;
  // Add other fields as needed
}

export const searchResumes = async (query: string): Promise<ApiResponse<SearchResult[]>> => {
  try {
    const response = await api.get('/resumes/search', {
      params: { query }
    });
    
    return { 
      success: true, 
      data: response.data 
    };
  } catch (error) {
    console.error('Error searching resumes:', error);
    const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
    return { 
      success: false, 
      error: errorMessage 
    };
  }
};
