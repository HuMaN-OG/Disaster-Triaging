import React from 'react';
import { UserRole } from '../types';
import { ShieldAlert, LayoutDashboard, Send, BarChart3, LogIn, LogOut, User } from 'lucide-react';

interface NavbarProps {
  activeTab: 'citizen' | 'dashboard' | 'analytics';
  setActiveTab: (tab: 'citizen' | 'dashboard' | 'analytics') => void;
  currentUser: { email: string; role: UserRole; full_name: string } | null;
  onOpenLogin: () => void;
  onLogout: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  currentUser,
  onOpenLogin,
  onLogout,
}) => {
  return (
    <header className="sticky top-0 z-40 bg-slate-950/90 backdrop-blur-md border-b border-slate-800 px-4 lg:px-8 py-3 flex items-center justify-between">
      {/* Brand */}
      <div className="flex items-center gap-3 cursor-pointer" onClick={() => setActiveTab('citizen')}>
        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-red-600 to-amber-600 flex items-center justify-center shadow-lg shadow-red-600/30">
          <ShieldAlert className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="text-base font-bold text-slate-100 tracking-tight leading-none">
            DISASTER<span className="text-red-500 font-extrabold ml-1">TRIAGE</span>
          </h1>
          <span className="text-[10px] font-mono text-slate-400 tracking-widest uppercase">
            AI Operational Triage System
          </span>
        </div>
      </div>

      {/* Mode Navigation Tabs */}
      <nav className="flex items-center gap-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800 text-xs">
        <button
          onClick={() => setActiveTab('citizen')}
          className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg font-medium transition-all ${
            activeTab === 'citizen'
              ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Send className="w-3.5 h-3.5" />
          <span>Citizen Portal</span>
        </button>

        <button
          onClick={() => setActiveTab('dashboard')}
          className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg font-medium transition-all ${
            activeTab === 'dashboard'
              ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <LayoutDashboard className="w-3.5 h-3.5" />
          <span>Rescue Command</span>
        </button>

        <button
          onClick={() => setActiveTab('analytics')}
          className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg font-medium transition-all ${
            activeTab === 'analytics'
              ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <BarChart3 className="w-3.5 h-3.5" />
          <span>Analytics</span>
        </button>
      </nav>

      {/* Auth Controls */}
      <div className="flex items-center gap-3">
        {currentUser ? (
          <div className="flex items-center gap-3">
            <div className="text-right hidden sm:block">
              <p className="text-xs font-semibold text-slate-200">{currentUser.full_name}</p>
              <span className={`text-[10px] font-mono px-2 py-0.5 rounded uppercase font-bold ${
                currentUser.role === 'Admin' ? 'bg-purple-950 text-purple-400 border border-purple-800' :
                currentUser.role === 'Responder' ? 'bg-amber-950 text-amber-400 border border-amber-800' : 'bg-slate-800 text-slate-300'
              }`}>
                {currentUser.role}
              </span>
            </div>

            <button
              onClick={onLogout}
              className="p-2 bg-slate-900 hover:bg-slate-800 text-slate-300 rounded-xl border border-slate-800 transition-colors flex items-center gap-1 text-xs"
              title="Sign Out"
            >
              <LogOut className="w-4 h-4 text-red-400" />
            </button>
          </div>
        ) : (
          <button
            onClick={onOpenLogin}
            className="flex items-center gap-1.5 px-3.5 py-2 bg-slate-900 hover:bg-slate-800 text-blue-400 border border-slate-800 rounded-xl text-xs font-semibold transition-colors"
          >
            <LogIn className="w-4 h-4" />
            <span>Responder Auth</span>
          </button>
        )}
      </div>
    </header>
  );
};
