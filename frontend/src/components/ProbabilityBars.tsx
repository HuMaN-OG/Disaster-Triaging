import React from 'react';

interface ProbabilityBarsProps {
  probabilities: Record<string, number>;
  predictedClass?: string;
}

export const ProbabilityBars: React.FC<ProbabilityBarsProps> = ({ probabilities, predictedClass }) => {
  if (!probabilities || Object.keys(probabilities).length === 0) {
    return <div className="text-slate-400 text-sm">No probability data available.</div>;
  }

  // Sort by probability descending
  const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="space-y-2.5">
      {sorted.map(([cls, prob]) => {
        const pct = Math.round(prob * 100);
        const isTop = cls === predictedClass || pct === Math.max(...Object.values(probabilities).map(v => Math.round(v * 100)));
        
        return (
          <div key={cls} className="space-y-1">
            <div className="flex justify-between text-xs font-mono font-medium">
              <span className={isTop ? "text-amber-400 font-bold" : "text-slate-300"}>
                {cls} {isTop && "★"}
              </span>
              <span className={isTop ? "text-amber-400 font-bold" : "text-slate-400"}>
                {pct}% ({prob.toFixed(4)})
              </span>
            </div>
            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden border border-slate-700/50">
              <div
                className={`h-full transition-all duration-500 rounded-full ${
                  isTop
                    ? 'bg-gradient-to-r from-amber-500 to-red-500 shadow-sm shadow-amber-500/50'
                    : 'bg-slate-600'
                }`}
                style={{ width: `${Math.max(pct, 2)}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
};
