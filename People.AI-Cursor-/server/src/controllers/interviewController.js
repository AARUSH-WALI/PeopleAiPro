import { supabase } from '../config/supabase.js';

// Get all interviews with pagination and filtering
export const getInterviews = async (req, res) => {
  try {
    const { 
      page = 1, 
      limit = 10, 
      status, 
      candidate_id,
      start_date,
      end_date
    } = req.query;
    
    const from = (page - 1) * limit;
    const to = from + parseInt(limit) - 1;

    let query = supabase
      .from('interviews')
      .select('*', { count: 'exact' })
      .order('date', { ascending: false });

    // Apply filters
    if (status) {
      query = query.eq('status', status);
    }
    if (candidate_id) {
      query = query.eq('candidate_id', candidate_id);
    }
    if (start_date) {
      query = query.gte('date', new Date(start_date).toISOString());
    }
    if (end_date) {
      // Add one day to include the end date
      const end = new Date(end_date);
      end.setDate(end.getDate() + 1);
      query = query.lt('date', end.toISOString());
    }

    // Apply pagination
    const { data, error, count } = await query.range(from, to);

    if (error) throw error;

    res.json({
      success: true,
      data: data || [],
      pagination: {
        total: count || 0,
        page: parseInt(page),
        limit: parseInt(limit),
        totalPages: Math.ceil((count || 0) / limit)
      }
    });
  } catch (error) {
    console.error('Error fetching interviews:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to fetch interviews',
      message: error.message 
    });
  }
};

// Get a specific interview by ID
export const getInterviewById = async (req, res) => {
  try {
    const { id } = req.params;
    const { data, error } = await supabase
      .from('interviews')
      .select('*')
      .eq('id', id)
      .single();

    if (error) {
      if (error.code === 'PGRST116') { // No rows returned
        return res.status(404).json({
          success: false,
          error: 'Interview not found'
        });
      }
      throw error;
    }

    if (!data) {
      return res.status(404).json({
        success: false,
        error: 'Interview not found'
      });
    }

    res.json({
      success: true,
      data: data
    });
  } catch (error) {
    console.error('Error fetching interview:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch interview',
      message: error.message
    });
  }
};

// Create a new interview
export const createInterview = async (req, res) => {
  try {
    const {
      title,
      description,
      candidateId,
      candidateName,
      candidateEmail,
      date,
      status = 'scheduled',
      interviewerId,
      interviewerName,
      interviewerEmail,
      meetingLink,
      meetingPlatform,
      interviewType,
      jobTitle,
      jobDescription
    } = req.body;

    // Validate required fields
    if (!title || !candidateId || !candidateName || !candidateEmail || !date) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields',
        required: ['title', 'candidateId', 'candidateName', 'candidateEmail', 'date']
      });
    }

    // Prepare interview data
    const interviewData = {
      title,
      description,
      candidate_id: candidateId,
      candidate_name: candidateName,
      candidate_email: candidateEmail,
      date: new Date(date).toISOString(),
      status,
      interviewer_id: interviewerId,
      interviewer_name: interviewerName,
      interviewer_email: interviewerEmail,
      meeting_link: meetingLink,
      meeting_platform: meetingPlatform,
      interview_type: interviewType,
      job_title: jobTitle,
      job_description: jobDescription,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };
    
    // Insert the new interview
    const { data, error } = await supabase
      .from('interviews')
      .insert([interviewData])
      .select('*');
      
    if (error) throw error;
    
    res.status(201).json({
      success: true,
      data: data[0]
    });
  } catch (error) {
    console.error('Error creating interview:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to create interview',
      message: error.message 
    });
  }
};

// Update an interview
export const updateInterview = async (req, res) => {
  try {
    const { id } = req.params;
    const {
      title,
      description,
      status,
      date,
      candidateId,
      candidateName,
      candidateEmail,
      interviewerId,
      interviewerName,
      interviewerEmail,
      meetingLink,
      meetingPlatform,
      interviewType,
      jobTitle,
      jobDescription,
      interviewerNotes,
      candidateFeedback,
      technicalAssessment,
      overallRating
    } = req.body;

    // Check if interview exists
    const { data: existingInterview, error: fetchError } = await supabase
      .from('interviews')
      .select('*')
      .eq('id', id)
      .single();

    if (fetchError || !existingInterview) {
      return res.status(404).json({
        success: false,
        error: 'Interview not found'
      });
    }

    // Prepare update data
    const updateData = {
      title: title !== undefined ? title : existingInterview.title,
      description: description !== undefined ? description : existingInterview.description,
      status: status !== undefined ? status : existingInterview.status,
      date: date !== undefined ? new Date(date).toISOString() : existingInterview.date,
      candidate_id: candidateId !== undefined ? candidateId : existingInterview.candidate_id,
      candidate_name: candidateName !== undefined ? candidateName : existingInterview.candidate_name,
      candidate_email: candidateEmail !== undefined ? candidateEmail : existingInterview.candidate_email,
      interviewer_id: interviewerId !== undefined ? interviewerId : existingInterview.interviewer_id,
      interviewer_name: interviewerName !== undefined ? interviewerName : existingInterview.interviewer_name,
      interviewer_email: interviewerEmail !== undefined ? interviewerEmail : existingInterview.interviewer_email,
      meeting_link: meetingLink !== undefined ? meetingLink : existingInterview.meeting_link,
      meeting_platform: meetingPlatform !== undefined ? meetingPlatform : existingInterview.meeting_platform,
      interview_type: interviewType !== undefined ? interviewType : existingInterview.interview_type,
      job_title: jobTitle !== undefined ? jobTitle : existingInterview.job_title,
      job_description: jobDescription !== undefined ? jobDescription : existingInterview.job_description,
      interviewer_notes: interviewerNotes !== undefined ? interviewerNotes : existingInterview.interviewer_notes,
      candidate_feedback: candidateFeedback !== undefined ? candidateFeedback : existingInterview.candidate_feedback,
      technical_assessment: technicalAssessment !== undefined ? technicalAssessment : existingInterview.technical_assessment,
      overall_rating: overallRating !== undefined ? overallRating : existingInterview.overall_rating,
      updated_at: new Date().toISOString()
    };
    
    // Update the interview
    const { data, error } = await supabase
      .from('interviews')
      .update(updateData)
      .eq('id', id)
      .select('*');
      
    if (error) throw error;
    
    if (!data || data.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Interview not found after update'
      });
    }
    
    res.json({
      success: true,
      data: data[0]
    });
  } catch (error) {
    console.error('Error updating interview:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to update interview',
      message: error.message 
    });
  }
};

// Delete an interview
export const deleteInterview = async (req, res) => {
  try {
    const { id } = req.params;
    
    // First check if the interview exists
    const { data: existingInterview, error: fetchError } = await supabase
      .from('interviews')
      .select('*')
      .eq('id', id)
      .single();

    if (fetchError || !existingInterview) {
      return res.status(404).json({ 
        success: false,
        error: 'Interview not found' 
      });
    }

    // Delete the interview
    const { error } = await supabase
      .from('interviews')
      .delete()
      .eq('id', id);

    if (error) throw error;

    res.json({
      success: true,
      message: 'Interview deleted successfully',
      data: {
        id: existingInterview.id,
        title: existingInterview.title,
        candidate_name: existingInterview.candidate_name
      }
    });
  } catch (error) {
    console.error('Error deleting interview:', error);
    res.status(500).json({ 
      success: false,
      error: 'Failed to delete interview',
      message: error.message 
    });
  }
};

// Update interview status
export const updateInterviewStatus = async (req, res) => {
try {
  const { id } = req.params;
  const { status } = req.body;

  if (!status) {
    return res.status(400).json({ 
      success: false,
      error: 'Status is required',
      validStatuses: ['scheduled', 'in_progress', 'completed', 'cancelled']
    });
  }

  // Validate status
  const validStatuses = ['scheduled', 'in_progress', 'completed', 'cancelled'];
  if (!validStatuses.includes(status)) {
    return res.status(400).json({
      success: false,
      error: 'Invalid status',
      message: `Status must be one of: ${validStatuses.join(', ')}`
    });
  }

  // Check if interview exists
  const { data: existingInterview, error: fetchError } = await supabase
    .from('interviews')
    .select('*')
    .eq('id', id)
    .single();

  if (fetchError || !existingInterview) {
    return res.status(404).json({
      success: false,
      error: 'Interview not found'
    });
  }

  // Update the status
  const { data, error } = await supabase
    .from('interviews')
    .update({ 
      status,
      // Update timestamps based on status
      ...(status === 'in_progress' && !existingInterview.actual_start_time 
        ? { actual_start_time: new Date().toISOString() } 
        : {}),
      ...(status === 'completed' && !existingInterview.actual_end_time 
        ? { actual_end_time: new Date().toISOString() } 
        : {}),
      updated_at: new Date().toISOString()
    })
    .eq('id', id)
    .select('*');

  if (error) throw error;
  
  if (!data || data.length === 0) {
    return res.status(404).json({
      success: false,
      error: 'Failed to update interview status'
    });
  }

  res.json({
    success: true,
    message: `Interview status updated to '${status}'`,
    data: data[0]
  });
} catch (error) {
  console.error('Error updating interview status:', error);
  res.status(500).json({ 
    success: false,
    error: 'Failed to update interview status',
    message: error.message 
  });
}
};
