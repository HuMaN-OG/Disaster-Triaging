import React from 'react';
import { AlertTriangle, CheckCircle2, ShieldAlert } from 'lucide-react';

interface RecommendationCardProps {
  recommendations: string[];
  priorityLevel: string;
  disasterType: string;
}

export const RecommendationCard: React.FC<RecommendationCardProps> = ({
  recommendations,
  priorityLevel,
  disasterType,
}) => {
  const getBadgeColor = (level: string) => {
    switch (level) {
      case 'Critical':
        return 'bg-red-950/80 text-red-400 border-red-800/60';
      case 'High':
        return 'bg-amber-950/80 text-amber-400 border-amber-800/60';
      case 'Medium':
        return 'bg-yellow-950/80 text-yellow-400 border-yellow-800/60';
      default:
        return 'bg-slate-800 text-slate-300 border-slate-700';
    }
  };

  return (
    <div className={`p-4 rounded-xl border ${getBadgeColor(priorityLevel)} space-y-3 shadow-lg`}>
      <div className="flex items-center justify-between border-b border-slate-700/40 pb-2">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-5 h-5 text-amber-400 animate-pulse" />
          <span className="font-semibold text-sm tracking-wide uppercase">
            Emergency Action Directives ({disasterType})
          </span>
        </div>
        <span className="text-xs px-2.5 py-0.5 rounded-full font-mono font-bold uppercase tracking-wider bg-slate-900/60 border border-current">
          {priorityLevel} Priority
        </span>
      </div>

      <ul className="space-y-2 text-xs">
        {recommendations.map((rec, i) => (
          <li key={i} className="flex items-start gap-2 text-slate-200">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span>{rec}</span>
          </li>
        ))}
      </ul>
    </div>
  );
};
