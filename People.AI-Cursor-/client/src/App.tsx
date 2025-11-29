
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import MainLayout from "./components/layout/MainLayout";
import Dashboard from "./pages/Dashboard";
import Users from "./pages/Users";
import Jobs from "./pages/Jobs";
import Interview from "./pages/Interview";
import Resume from "./pages/Resume";
import NotFound from "./pages/NotFound";
import PersonalityTest from "./components/PersonalityTest";
import Settings from "./pages/Settings";
// import RecentActivity from "./components/dashboard/RecentActivity";
// import CandidateStatus from "./components/dashboard/CandidateStatus";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={
            <MainLayout>
              <Dashboard />
            </MainLayout>
          } />
          <Route path="/users" element={
            <MainLayout>
              <Users />
            </MainLayout>
          } />
          <Route path="/jobs" element={
            <MainLayout>
              <Jobs />
            </MainLayout>
          } />
          <Route path="/interview" element={
            <MainLayout>
              <Interview />
            </MainLayout>
          } />
          <Route path="/resume" element={
            <MainLayout>
              <Resume />
            </MainLayout>
          } />
          <Route path="/settings" element={
            <MainLayout>
              <Settings />
            </MainLayout>
          } />
          {/* <Route path="/recent-activity" element={
            <MainLayout>
              <RecentActivity />
            </MainLayout>
          } /> */}
          {/* <Route path="/candidate-status" element={
            <MainLayout>
              <CandidateStatus approvedCount={45} reviewCount={30} rejectedCount={25} />
            </MainLayout>
          } /> */}
          <Route path="/personality-test" element={<PersonalityTest />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
