import React from 'react';
import { Activity } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="bg-slate-950 border-t border-slate-900 py-6 px-4 lg:px-8 text-xs text-slate-500 flex flex-col sm:flex-row items-center justify-between gap-4">
      <div className="flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
        <span className="font-mono text-slate-400">System Telemetry Online & Operational</span>
      </div>

      <div className="flex items-center gap-6 font-mono text-[11px]">
        <span>6-Class Ensemble Vision</span>
        <span>•</span>
        <span>CatBoost Text Priority</span>
        <span>•</span>
        <span>FastAPI & SQLite</span>
      </div>
    </footer>
  );
};
