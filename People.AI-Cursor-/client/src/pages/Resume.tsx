
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ResumeUpload from "@/components/resume/ResumeUpload";
import ResumeForm from "@/components/resume/ResumeForm";
import { ResumeData } from "@/types/resume";

export default function Resume() {
  const [activeTab, setActiveTab] = useState("upload");
  const [resumeData, setResumeData] = useState<ResumeData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [parsedFile, setParsedFile] = useState<File | null>(null);
  const [jsonData, setJsonData] = useState<string>("");

  const handleResumeUploaded = (data: ResumeData, file: File) => {
    setResumeData(data);
    setParsedFile(file);
    setActiveTab("form");
    
    // Convert the parsed data to a formatted JSON string
    setJsonData(JSON.stringify(data, null, 2));
  };

  const handleParsing = (loading: boolean) => {
    setIsLoading(loading);
  };

  return (
    <div className="page-container p-6 bg-gradient-to-br from-purple-50 to-white">
      <h1 className="text-2xl font-bold mb-6 text-purple-800">Resume Parser</h1>
      
      <Card className="shadow-lg border-purple-200">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2 bg-purple-100">
            <TabsTrigger 
              value="upload" 
              disabled={isLoading}
              className="text-base py-3 data-[state=active]:bg-purple-600 data-[state=active]:text-white"
            >
              Upload Resume
            </TabsTrigger>
            <TabsTrigger 
              value="form" 
              disabled={!resumeData || isLoading} 
              className="text-base py-3 data-[state=active]:bg-purple-600 data-[state=active]:text-white"
            >
              Review & Edit
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="upload" className="p-6 bg-white">
            <ResumeUpload 
              onResumeUploaded={handleResumeUploaded} 
              onParsingStateChange={handleParsing}
            />
          </TabsContent>
          
          <TabsContent value="form" className="p-6 bg-gradient-to-br from-purple-50 to-white">
            {resumeData ? (
              <ResumeForm 
                resumeData={resumeData} 
                setResumeData={setResumeData} 
                parsedFile={parsedFile}
                jsonData={jsonData}
                setJsonData={setJsonData}
              />
            ) : (
              <div className="text-center py-10">
                <p className="text-gray-500">Please upload a resume first</p>
                <Button 
                  variant="outline" 
                  className="mt-4 border-purple-200 hover:bg-purple-50 hover:text-purple-700"
                  onClick={() => setActiveTab("upload")}
                >
                  Go to Upload
                </Button>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </Card>
    </div>
  );
}
