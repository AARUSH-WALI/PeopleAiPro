// Environment types
type Environment = 'local' | 'development' | 'production';

// API configuration interface
interface ApiConfig {
  baseUrl: string;
  endpoints: {
    users: string;
    criteria: string;
    // Add other endpoints here as needed
  };
}

// Environment detection
const getEnvironment = (): Environment => {
  const hostname = window.location.hostname;
  
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'local';
  }
  
  if (hostname.includes('dev.') || hostname.includes('staging.')) {
    return 'development';
  }
  
  return 'production';
};

// Environment-specific configurations
const configs: Record<Environment, ApiConfig> = {
  local: {
    baseUrl: 'http://localhost:5000/api',
    endpoints: {
      users: '/resumes',
      criteria: '/criteria',
    },
  },
  development: {
    baseUrl: 'https://people-ai-new.onrender.com/api',
    endpoints: {
      users: '/resumes',
      criteria: '/criteria',
    },
  },
  production: {
    baseUrl: 'https://people-ai-new.onrender.com/api',
    endpoints: {
      users: '/resumes',
      criteria: '/criteria',
    },
  },
};

// Get current environment config
const env = getEnvironment();
const apiConfig = configs[env];

// Helper function to get full URL
const getApiUrl = (endpoint: keyof ApiConfig['endpoints']): string => {
  return `${apiConfig.baseUrl}${apiConfig.endpoints[endpoint]}`;
};

export { apiConfig, getApiUrl, env };
