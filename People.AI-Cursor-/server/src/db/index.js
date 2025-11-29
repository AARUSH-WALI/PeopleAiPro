import { createClient } from '@supabase/supabase-js';
import dotenv from 'dotenv';

dotenv.config();

// Initialize Supabase client
export const supabase = createClient(
  process.env.SUPABASE_URL || '',
  process.env.SUPABASE_SERVICE_KEY || '',
  {
    auth: {
      autoRefreshToken: false,
      persistSession: false
    },
    db: {
      schema: 'public'
    }
  }
);

// Helper function to execute raw SQL queries
export const query = async (sql, params = []) => {
  try {
    const { data, error } = await supabase.rpc('execute_sql', {
      query: sql,
      params: params
    });
    
    if (error) throw error;
    return { rows: data || [], rowCount: data?.length || 0 };
  } catch (error) {
    console.error('Database query error:', error);
    throw error;
  }
};

export default {
  supabase,
  query
};
