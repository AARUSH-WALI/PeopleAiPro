import express from 'express';
import { supabase } from '../db/index.js';

const router = express.Router();

// Get current criteria
router.get('/', async (req, res) => {
  try {
    const { data, error } = await supabase
      .from('fitment_criteria')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(1);

    if (error) throw error;

    // If no criteria exists yet, return default values
    if (!data || data.length === 0) {
      return res.status(200).json({
        id: null,
        best_fit: 80,
        average_fit: 50,
        not_fit: 0,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      });
    }

    res.json(data[0]);
  } catch (error) {
    console.error('Error fetching criteria:', error);
    res.status(500).json({ error: 'Failed to fetch criteria', details: error.message });
  }
});

// Update criteria
router.put('/', async (req, res) => {
  const { best_fit, average_fit, not_fit } = req.body;

  // Validate input
  if (best_fit === undefined || average_fit === undefined || not_fit === undefined) {
    return res.status(400).json({ error: 'Missing required fields' });
  }

  // Validate the criteria values
  if (best_fit <= average_fit || average_fit <= not_fit || not_fit < 0) {
    return res.status(400).json({ 
      error: 'Invalid criteria values. Must satisfy: best_fit > average_fit > not_fit >= 0' 
    });
  }

  try {
    // First, get the current criteria to check if it exists
    const { data: current, error: fetchError } = await supabase
      .from('fitment_criteria')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(1);

    if (fetchError) throw fetchError;

    let result;
    if (current && current.length > 0) {
      // Update existing criteria
      const { data: updated, error: updateError } = await supabase
        .from('fitment_criteria')
        .update({
          best_fit,
          average_fit,
          not_fit,
          updated_at: new Date().toISOString()
        })
        .eq('id', current[0].id)
        .select();

      if (updateError) throw updateError;
      result = updated[0];
    } else {
      // Insert new criteria if none exists
      const { data: inserted, error: insertError } = await supabase
        .from('fitment_criteria')
        .insert([
          {
            best_fit,
            average_fit,
            not_fit
          }
        ])
        .select();

      if (insertError) throw insertError;
      result = inserted[0];
    }
    
    res.json(result);
  } catch (error) {
    console.error('Error updating criteria:', error);
    res.status(500).json({ 
      error: 'Failed to update criteria',
      details: error.message 
    });
  }
});

export default router;
