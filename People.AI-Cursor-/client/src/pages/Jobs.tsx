import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { RefreshCw, Plus, ChevronLeft } from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import axios from "axios";
import { toast } from "sonner";

const API_BASE_URL = import.meta.env.PROD
  ? import.meta.env.VITE_PROD_API_BASE_URL
  : import.meta.env.VITE_DEV_API_BASE_URL || 'http://localhost:5000/api';

interface Job {
  id: string;
  title: string;
  active: boolean;
  description?: string;
  requirements?: string[];
  created_at?: string;
  updated_at?: string;
}

export default function Jobs() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [showNewJobDialog, setShowNewJobDialog] = useState(false);
  const [showViewDialog, setShowViewDialog] = useState(false);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [newJob, setNewJob] = useState({
    title: '',
    description: '',
    requirements: ''
  });

  // Fetch jobs from API
  const fetchJobs = async () => {
    try {
      setRefreshing(true);
      const response = await axios.get(`${API_BASE_URL}/jobs`);
      const jobsData = response.data?.data || [];

      // Transform data to match our interface
      const transformedJobs = jobsData.map((job: any) => ({
        id: job.id,
        title: job.title,
        active: job.active ?? true,
        description: job.description || '',
        requirements: Array.isArray(job.requirements) ? job.requirements : [],
        created_at: job.created_at,
        updated_at: job.updated_at
      }));

      setJobs(transformedJobs);
      console.log('[Jobs] Fetched jobs:', transformedJobs);
    } catch (error: any) {
      console.error('[Jobs] Error fetching jobs:', error);
      toast.error('Failed to load jobs: ' + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  // Load jobs on component mount
  useEffect(() => {
    fetchJobs();
  }, []);

  // Handle refresh
  const handleRefresh = () => {
    fetchJobs();
  };

  // Handle submit new job
  const handleSubmitJob = async () => {
    if (!newJob.title.trim()) {
      toast.error('Job title is required');
      return;
    }

    try {
      const response = await axios.post(`${API_BASE_URL}/jobs`, {
        title: newJob.title,
        description: newJob.description || null,
        requirements: newJob.requirements || null,
        active: true
      });

      if (response.data.success) {
        toast.success('Job created successfully');
        setShowNewJobDialog(false);
        setNewJob({ title: '', description: '', requirements: '' });
        fetchJobs(); // Refresh the list
      }
    } catch (error: any) {
      console.error('[Jobs] Error creating job:', error);
      toast.error('Failed to create job: ' + (error.response?.data?.error || error.message));
    }
  };

  // Handle toggle active status
  const handleToggleActive = async (id: string) => {
    const job = jobs.find(j => j.id === id);
    if (!job) return;

    const newActiveStatus = !job.active;

    try {
      const response = await axios.put(`${API_BASE_URL}/jobs/${id}`, {
        active: newActiveStatus
      });

      if (response.data.success) {
        // Update local state
        setJobs(jobs.map(j =>
          j.id === id ? { ...j, active: newActiveStatus } : j
        ));
        toast.success('Job status updated');
      }
    } catch (error: any) {
      console.error('[Jobs] Error toggling job status:', error);
      toast.error('Failed to update job status: ' + (error.response?.data?.error || error.message));
      // Revert on error
      fetchJobs();
    }
  };

  // Handle view job
  const handleViewJob = async (job: Job) => {
    try {
      // Fetch full job details
      const response = await axios.get(`${API_BASE_URL}/jobs/${job.id}`);
      if (response.data.success) {
        const fullJob = response.data.data;
        setSelectedJob({
          id: fullJob.id,
          title: fullJob.title,
          active: fullJob.active,
          description: fullJob.description || '',
          requirements: Array.isArray(fullJob.requirements) ? fullJob.requirements : [],
          created_at: fullJob.created_at,
          updated_at: fullJob.updated_at
        });
        setShowViewDialog(true);
        setIsEditing(false);
      }
    } catch (error: any) {
      console.error('[Jobs] Error fetching job details:', error);
      // Fallback to local data
      setSelectedJob(job);
      setShowViewDialog(true);
      setIsEditing(false);
    }
  };

  // Handle edit
  const handleEdit = () => {
    setIsEditing(true);
  };

  // Handle save edit
  const handleSaveEdit = async () => {
    if (!selectedJob) return;

    if (!selectedJob.title.trim()) {
      toast.error('Job title is required');
      return;
    }

    try {
      const response = await axios.put(`${API_BASE_URL}/jobs/${selectedJob.id}`, {
        title: selectedJob.title,
        description: selectedJob.description || null,
        requirements: selectedJob.requirements || null,
        active: selectedJob.active
      });

      if (response.data.success) {
        toast.success('Job updated successfully');
        setIsEditing(false);
        fetchJobs(); // Refresh the list
      }
    } catch (error: any) {
      console.error('[Jobs] Error updating job:', error);
      toast.error('Failed to update job: ' + (error.response?.data?.error || error.message));
    }
  };

  // Handle delete
  const handleDelete = async () => {
    if (!selectedJob) return;

    if (!confirm(`Are you sure you want to delete "${selectedJob.title}"?`)) {
      return;
    }

    try {
      const response = await axios.delete(`${API_BASE_URL}/jobs/${selectedJob.id}`);

      if (response.data.success) {
        toast.success('Job deleted successfully');
        setShowViewDialog(false);
        setSelectedJob(null);
        fetchJobs(); // Refresh the list
      }
    } catch (error: any) {
      console.error('[Jobs] Error deleting job:', error);
      toast.error('Failed to delete job: ' + (error.response?.data?.error || error.message));
    }
  };

  // Handle delete from table
  const handleDeleteFromTable = async (job: Job) => {
    if (!confirm(`Are you sure you want to delete "${job.title}"?`)) {
      return;
    }

    try {
      const response = await axios.delete(`${API_BASE_URL}/jobs/${job.id}`);

      if (response.data.success) {
        toast.success('Job deleted successfully');
        fetchJobs(); // Refresh the list
      }
    } catch (error: any) {
      console.error('[Jobs] Error deleting job:', error);
      toast.error('Failed to delete job: ' + (error.response?.data?.error || error.message));
    }
  };

  if (loading) {
    return (
      <div className="page-container p-4 bg-gray-100 min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading jobs...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container p-4 bg-gray-100 min-h-screen">
      <div className="bg-white rounded-2xl shadow-md hover:shadow-lg transition-shadow duration-300 overflow-hidden border border-gray-200">
        <div className="p-4 flex justify-between items-center border-b border-gray-200 bg-gradient-to-r from-blue-50 via-white to-blue-50">
          <h1 className="text-2xl font-bold text-gray-800">Job Listings</h1>
          <Button
            size="icon"
            variant="outline"
            className="hover:bg-blue-100 transition-colors duration-200"
            onClick={handleRefresh}
            disabled={refreshing}
          >
            <RefreshCw size={16} className={`text-gray-700 ${refreshing ? 'animate-spin' : ''}`} />
          </Button>
        </div>

        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Title</TableHead>
                <TableHead>Action</TableHead>
                <TableHead className="text-right">Active</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={3} className="text-center py-8 text-gray-500">
                    No jobs found. Click "Add New Job" to create one.
                  </TableCell>
                </TableRow>
              ) : (
                jobs.map((job) => (
                  <TableRow key={job.id}>
                    <TableCell className="font-medium">{job.title}</TableCell>
                    <TableCell>
                      <div className="flex space-x-2">
                        <Button
                          size="sm"
                          className="bg-blue-500 hover:bg-blue-600"
                          onClick={() => handleViewJob(job)}
                        >
                          View
                        </Button>
                        <Button
                          size="sm"
                          variant="destructive"
                          onClick={() => handleDeleteFromTable(job)}
                        >
                          Delete
                        </Button>
                      </div>
                    </TableCell>
                    <TableCell className="text-right">
                      <Switch
                        checked={job.active}
                        onCheckedChange={() => handleToggleActive(job.id)}
                      />
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>

        <div className="p-4">
          <Button
            className="bg-blue-500 hover:bg-blue-600"
            onClick={() => setShowNewJobDialog(true)}
          >
            <Plus className="h-4 w-4 mr-2" />
            Add New Job
          </Button>
        </div>
      </div>

      {/* Add New Job Dialog */}
      <Dialog open={showNewJobDialog} onOpenChange={setShowNewJobDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Add New Job</DialogTitle>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Job Title *</label>
              <Input
                placeholder="Job Title"
                value={newJob.title}
                onChange={(e) => setNewJob({ ...newJob, title: e.target.value })}
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Job Description</label>
              <Textarea
                placeholder="Job Description"
                value={newJob.description}
                onChange={(e) => setNewJob({ ...newJob, description: e.target.value })}
                className="min-h-[100px]"
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">Job Requirements (one per line)</label>
              <Textarea
                placeholder="Enter requirements, one per line"
                value={newJob.requirements}
                onChange={(e) => setNewJob({ ...newJob, requirements: e.target.value })}
                className="min-h-[100px]"
              />
            </div>

            <div className="flex justify-end space-x-2 pt-4">
              <Button variant="outline" onClick={() => setShowNewJobDialog(false)}>
                Cancel
              </Button>
              <Button
                className="bg-blue-500 hover:bg-blue-600"
                onClick={handleSubmitJob}
              >
                Add Job
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* View/Edit Job Dialog */}
      <Dialog open={showViewDialog} onOpenChange={setShowViewDialog}>
        <DialogContent className="max-w-2xl max-h-[85vh] overflow-hidden flex flex-col">
          <DialogHeader className="flex-shrink-0">
            <DialogTitle className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => {
                  setShowViewDialog(false);
                  setSelectedJob(null);
                  setIsEditing(false);
                }}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <span className="text-xl">{selectedJob?.title}</span>
            </DialogTitle>
          </DialogHeader>

          {selectedJob && (
            <div className="space-y-4 flex-1 overflow-y-auto pr-2">
              {isEditing && (
                <div>
                  <label className="text-sm font-medium mb-2 block">Job Title *</label>
                  <Input
                    value={selectedJob.title}
                    onChange={(e) => setSelectedJob({
                      ...selectedJob,
                      title: e.target.value
                    })}
                  />
                </div>
              )}

              <div>
                <h3 className="text-base font-semibold mb-2">Description</h3>
                {isEditing ? (
                  <Textarea
                    value={selectedJob.description || ''}
                    onChange={(e) => setSelectedJob({
                      ...selectedJob,
                      description: e.target.value
                    })}
                    className="min-h-[150px] max-h-[200px] resize-y text-sm"
                  />
                ) : (
                  <ScrollArea className="h-[200px] border border-gray-200 rounded-md p-3">
                    <div className="whitespace-pre-line text-gray-700 text-sm">
                      {selectedJob.description || 'No description provided'}
                    </div>
                  </ScrollArea>
                )}
              </div>

              <div>
                <h3 className="text-base font-semibold mb-2">Requirements</h3>
                {isEditing ? (
                  <Textarea
                    value={selectedJob.requirements?.join("\n") || ''}
                    onChange={(e) => setSelectedJob({
                      ...selectedJob,
                      requirements: e.target.value.split("\n").filter(r => r.trim())
                    })}
                    className="min-h-[150px] max-h-[200px] resize-y text-sm"
                    placeholder="Enter requirements, one per line"
                  />
                ) : (
                  <ScrollArea className="h-[200px] border border-gray-200 rounded-md p-3">
                    <ul className="list-disc pl-5 space-y-1">
                      {selectedJob.requirements && selectedJob.requirements.length > 0 ? (
                        selectedJob.requirements.map((req, index) => (
                          <li key={index} className="text-gray-700 text-sm">{req}</li>
                        ))
                      ) : (
                        <li className="text-gray-500 italic text-sm">No requirements specified</li>
                      )}
                    </ul>
                  </ScrollArea>
                )}
              </div>

            </div>
          )}
          <div className="flex justify-end space-x-2 pt-4 border-t flex-shrink-0">
            {selectedJob && (
              <>
                {isEditing ? (
                  <>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setIsEditing(false)}
                    >
                      Cancel
                    </Button>
                    <Button
                      className="bg-blue-500 hover:bg-blue-600"
                      size="sm"
                      onClick={handleSaveEdit}
                    >
                      Save Changes
                    </Button>
                  </>
                ) : (
                  <>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleEdit}
                    >
                      Edit
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={handleDelete}
                    >
                      Delete
                    </Button>
                  </>
                )}
              </>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
