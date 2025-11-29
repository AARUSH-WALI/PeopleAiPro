
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";

interface Employee {
  name: string;
  role: string;
  fitment: string;
}

interface JobFitmentTableProps {
  jobRoles: string[];
  fitCategories: string[];
  employees: Employee[];
}

export default function JobFitmentTable({ jobRoles, fitCategories, employees }: JobFitmentTableProps) {
  const [selectedRole, setSelectedRole] = useState<string>("all");
  const [selectedFitment, setSelectedFitment] = useState<string>("all");
  const [showAll, setShowAll] = useState(false);

  // Update the fitment for specific employees
  const updatedEmployees = employees.map((employee) => {
    if (employee.name === "Antra Patel") {
      return { ...employee, fitment: "Mid Fit" };
    }
    if (employee.name === "Aarush Wali") {
      return { ...employee, fitment: "Best Fit" };
    }
    return employee;
  });

  const filteredEmployees = updatedEmployees.filter(employee => {
    const roleMatch = selectedRole === "all" || employee.role === selectedRole;
    const fitMatch = selectedFitment === "all" || employee.fitment === selectedFitment;
    return roleMatch && fitMatch;
  });
  
  // Display only 7 employees by default (updated from 5 to 7)
  const displayedEmployees = showAll ? filteredEmployees : filteredEmployees.slice(0, 7);

  return (
    <div className="bg-white rounded-lg p-6 shadow-2xl h-full flex flex-col">
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h3 className="text-xl font-semibold text-indigo-600">Job Fitment</h3>
          {filteredEmployees.length > 7 && (
            <Button variant="ghost" size="sm" onClick={() => setShowAll(!showAll)} className="flex items-center gap-2">
              {showAll ? "Show Less" : "View All"}
              <ArrowRight size={16} />
            </Button>
          )}
        </div>
        <div className="flex flex-col space-y-3">
          <Select value={selectedRole} onValueChange={setSelectedRole}>
            <SelectTrigger className="w-full border-2 border-indigo-600 rounded-lg focus:ring-indigo-500">
              <SelectValue placeholder="Select Role" className="font-semibold text-gray-700" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all" className="text-indigo-600">All Roles</SelectItem>
              {jobRoles.map((role) => (
                <SelectItem key={role} value={role} className="text-gray-700">
                  {role}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select value={selectedFitment} onValueChange={setSelectedFitment}>
            <SelectTrigger className="w-full border-2 border-indigo-600 rounded-lg focus:ring-indigo-500">
              <SelectValue placeholder="Select Fitment" className="font-semibold text-gray-700" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all" className="text-indigo-600">All Categories</SelectItem>
              {fitCategories.map((category) => (
                <SelectItem key={category} value={category} className="text-gray-700">
                  {category}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="overflow-x-auto mt-4 flex-grow">
        <table className="min-w-full border-separate border-spacing-0 rounded-lg shadow-lg">
          <thead>
            <tr className="bg-indigo-500 text-white">
              <th className="py-2 px-4 text-left text-sm font-semibold rounded-tl-lg">Name</th>
              <th className="py-2 px-4 text-left text-sm font-semibold">Role</th>
              <th className="py-2 px-4 text-left text-sm font-semibold rounded-tr-lg">Fitment</th>
            </tr>
          </thead>
          <tbody className="bg-white">
            {displayedEmployees.map((employee, index) => (
              <tr key={index} className="hover:bg-indigo-100 transition-colors">
                <td className="py-2 px-4 border-t border-gray-100 text-sm font-medium text-gray-800">{employee.name}</td>
                <td className="py-2 px-4 border-t border-gray-100 text-sm font-medium text-gray-600">{employee.role}</td>
                <td className="py-2 px-4 border-t border-gray-100 text-sm font-medium">
                  <span className={`inline-flex items-center px-4 py-1 rounded-full text-sm font-medium ${
                    employee.fitment === "Best Fit" ? "bg-green-500 text-white" :
                    employee.fitment === "Mid Fit" ? "bg-yellow-500 text-white" :
                    "bg-red-500 text-white"
                  }`}>
                    {employee.fitment}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
