
import { useState } from "react";
import { format, formatDistanceToNow } from "date-fns";

type CandidateActivityStatus = "Approved" | "Reviewing" | "Rejected";

interface CandidateActivity {
  id: string;
  name: string;
  title: string;
  position: string;
  department: string;
  status: CandidateActivityStatus;
  updatedAt: Date;
}

export default function RecentActivity() {
  const [activities] = useState<CandidateActivity[]>([
    {
      id: "1",
      name: "Dr. Jane Smith",
      title: "Dr.",
      position: "Assistant Professor",
      department: "Computer Science",
      status: "Approved",
      updatedAt: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
    },
    {
      id: "2",
      name: "Dr. Michael Johnson",
      title: "Dr.",
      position: "Associate Professor",
      department: "Physics",
      status: "Reviewing",
      updatedAt: new Date(Date.now() - 5 * 60 * 60 * 1000), // 5 hours ago
    },
    {
      id: "3",
      name: "Dr. Sarah Williams",
      title: "Dr.",
      position: "Professor",
      department: "Mathematics",
      status: "Rejected",
      updatedAt: new Date(Date.now() - 24 * 60 * 60 * 1000), // 1 day ago
    },
    {
      id: "4",
      name: "Dr. Robert Brown",
      title: "Dr.",
      position: "Assistant Professor",
      department: "Chemistry",
      status: "Reviewing",
      updatedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000), // 2 days ago
    },
    {
      id: "5",
      name: "Dr. Emily Davis",
      title: "Dr.",
      position: "Associate Professor",
      department: "Biology",
      status: "Approved",
      updatedAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000), // 3 days ago
    },
  ]);

  const getStatusBadgeClasses = (status: CandidateActivityStatus) => {
    switch (status) {
      case "Approved":
        return "bg-green-100 text-green-600 border-green-300";
      case "Reviewing":
        return "bg-purple-100 text-purple-600 border-purple-300";
      case "Rejected":
        return "bg-red-100 text-red-600 border-red-300";
      default:
        return "bg-gray-100 text-gray-600 border-gray-200";
    }
  };

  const getFormattedTime = (date: Date) => {
    const distance = formatDistanceToNow(date, { addSuffix: false });
    if (distance.includes("day")) {
      if (distance.includes("1 day")) {
        return "1 day ago";
      }
      return `${parseInt(distance)} days ago`;
    }
    return `${parseInt(distance)} hours ago`;
  };

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden border border-gray-200">
      <div className="p-5 flex justify-between items-center border-b border-gray-200 bg-gradient-to-r from-purple-50 to-white">
        <h2 className="text-xl font-bold text-gray-800">Recent Activity</h2>
        <button className="text-purple-600 hover:text-purple-800 text-sm font-medium">
          View All
        </button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-gray-50 text-gray-600 border-b border-gray-200">
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                Candidate
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                Position
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                Department
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium uppercase tracking-wider">
                Time
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {activities.map((activity) => (
              <tr key={activity.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-800">
                  {activity.name}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                  {activity.position}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                  {activity.department}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span
                    className={`px-3 py-1 inline-flex text-xs leading-5 font-semibold rounded-full border ${getStatusBadgeClasses(
                      activity.status
                    )}`}
                  >
                    {activity.status === "Reviewing" ? "Reviewing" : activity.status}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 text-right">
                  {getFormattedTime(activity.updatedAt)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
