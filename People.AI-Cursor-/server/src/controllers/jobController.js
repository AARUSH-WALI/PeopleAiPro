import { supabase } from '../config/supabase.js';

// Get all jobs
export const getJobs = async (req, res) => {
  try {
    const { active } = req.query;

    let query = supabase
      .from('jobs')
      .select('*')
      .order('created_at', { ascending: false });

    // Filter by active status if provided
    if (active !== undefined) {
      query = query.eq('active', active === 'true');
    }

    const { data, error, count } = await query;

    if (error) throw error;

    res.json({
      success: true,
      data: data || [],
      total: count || data?.length || 0
    });
  } catch (error) {
    console.error('Error getting jobs:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

// Get a single job by ID
export const getJobById = async (req, res) => {
  try {
    const { id } = req.params;

    const { data, error } = await supabase
      .from('jobs')
      .select('*')
      .eq('id', id)
      .single();

    if (error) throw error;
    if (!data) {
      return res.status(404).json({
        success: false,
        error: 'Job not found'
      });
    }

    res.json({
      success: true,
      data
    });
  } catch (error) {
    console.error('Error getting job:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

// Create a new job
export const createJob = async (req, res) => {
  try {
    const { title, description, requirements, active = true } = req.body;

    // Validate required fields
    if (!title || title.trim() === '') {
      return res.status(400).json({
        success: false,
        error: 'Job title is required'
      });
    }

    // Process requirements - convert string to array if needed
    let requirementsArray = [];
    if (requirements) {
      if (typeof requirements === 'string') {
        // Split by newlines and filter empty strings
        requirementsArray = requirements
          .split('\n')
          .map(r => r.trim())
          .filter(r => r.length > 0);
      } else if (Array.isArray(requirements)) {
        requirementsArray = requirements;
      }
    }

    const jobData = {
      title: title.trim(),
      description: description?.trim() || null,
      requirements: requirementsArray.length > 0 ? requirementsArray : null,
      active: active === true || active === 'true'
    };

    const { data, error } = await supabase
      .from('jobs')
      .insert([jobData])
      .select()
      .single();

    if (error) throw error;

    res.status(201).json({
      success: true,
      message: 'Job created successfully',
      data
    });
  } catch (error) {
    console.error('Error creating job:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

// Update a job
export const updateJob = async (req, res) => {
  try {
    const { id } = req.params;
    const { title, description, requirements, active } = req.body;

    if (!id) {
      return res.status(400).json({
        success: false,
        error: 'Job ID is required'
      });
    }

    const updateData = {};

    if (title !== undefined) {
      if (!title || title.trim() === '') {
        return res.status(400).json({
          success: false,
          error: 'Job title cannot be empty'
        });
      }
      updateData.title = title.trim();
    }

    if (description !== undefined) {
      updateData.description = description?.trim() || null;
    }

    if (requirements !== undefined) {
      let requirementsArray = [];
      if (requirements) {
        if (typeof requirements === 'string') {
          requirementsArray = requirements
            .split('\n')
            .map(r => r.trim())
            .filter(r => r.length > 0);
        } else if (Array.isArray(requirements)) {
          requirementsArray = requirements;
        }
      }
      updateData.requirements = requirementsArray.length > 0 ? requirementsArray : null;
    }

    if (active !== undefined) {
      updateData.active = active === true || active === 'true';
    }

    updateData.updated_at = new Date().toISOString();

    const { data, error } = await supabase
      .from('jobs')
      .update(updateData)
      .eq('id', id)
      .select()
      .single();

    if (error) throw error;
    if (!data) {
      return res.status(404).json({
        success: false,
        error: 'Job not found'
      });
    }

    res.json({
      success: true,
      message: 'Job updated successfully',
      data
    });
  } catch (error) {
    console.error('Error updating job:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

// Delete a job
export const deleteJob = async (req, res) => {
  try {
    const { id } = req.params;

    if (!id) {
      return res.status(400).json({
        success: false,
        error: 'Job ID is required'
      });
    }

    const { error } = await supabase
      .from('jobs')
      .delete()
      .eq('id', id);

    if (error) throw error;

    res.json({
      success: true,
      message: 'Job deleted successfully'
    });
  } catch (error) {
    console.error('Error deleting job:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

