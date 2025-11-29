import { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface StatCardProps {
  value: string | number;
  title: string;
  icon?: ReactNode;
  color?: "purple" | "blue" | "white" | "green" | "rose";
  className?: string;
  children?: ReactNode;
  onClick?: () => void;
}

export default function StatCard({ 
  value, 
  title, 
  icon, 
  color = "white",
  className,
  children,
  onClick
}: StatCardProps) {

  const bgColor = {
    purple: "bg-purple-600",
    blue: "bg-blue-600",
    green: "bg-green-600",
    rose: "bg-teal-500",
    white: "bg-white"
  }[color];

  return (
    <div 
      className={cn(
        "stat-card flex items-start justify-between p-6 rounded-xl shadow-md relative text-white",
        bgColor,
        onClick && "cursor-pointer hover:bg-opacity-90 transition-colors",
        className
      )}
      onClick={onClick}
    >
      <div>
        <div className="text-3xl font-bold">{value}</div>
        <div className="text-lg font-bold mt-1">{title}</div>
      </div>
      {icon && (
        <div className="p-3 rounded-full bg-white/20">
          {icon}
        </div>
      )}
      {children}
    </div>
  );
}
