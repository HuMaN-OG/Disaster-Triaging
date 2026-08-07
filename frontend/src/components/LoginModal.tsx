import React, { useState } from 'react';
import { loginApi, registerApi } from '../api/auth';
import { User, UserRole } from '../types';
import { X, Lock, Mail, User as UserIcon, ShieldAlert } from 'lucide-react';

interface LoginModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLoginSuccess: (user: { email: string; role: UserRole; full_name: string }) => void;
}

export const LoginModal: React.FC<LoginModalProps> = ({ isOpen, onClose, onLoginSuccess }) => {
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [role, setRole] = useState<UserRole>('Responder');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (isRegister) {
        await registerApi({ email, password, full_name: fullName, role });
        // Automatically login after register
        const tokenData = await loginApi(email, password);
        localStorage.setItem('disaster_triage_token', tokenData.access_token);
        onLoginSuccess({ email, role: tokenData.role, full_name: tokenData.full_name });
      } else {
        const tokenData = await loginApi(email, password);
        localStorage.setItem('disaster_triage_token', tokenData.access_token);
        onLoginSuccess({ email, role: tokenData.role, full_name: tokenData.full_name });
      }
      onClose();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Authentication failed. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  const handleQuickLogin = async (quickEmail: string, quickPass: string) => {
    setEmail(quickEmail);
    setPassword(quickPass);
    setLoading(true);
    setError(null);
    try {
      const tokenData = await loginApi(quickEmail, quickPass);
      localStorage.setItem('disaster_triage_token', tokenData.access_token);
      onLoginSuccess({ email: quickEmail, role: tokenData.role, full_name: tokenData.full_name });
      onClose();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Quick login failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm">
      <div className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl p-6 relative">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2 text-blue-500 mb-2">
          <ShieldAlert className="w-6 h-6 text-amber-500" />
          <h2 className="text-xl font-bold text-slate-100">
            {isRegister ? 'Register Account' : 'Responder & Command Login'}
          </h2>
        </div>
        <p className="text-xs text-slate-400 mb-6">
          Access rescue dispatch actions and real-time operations dashboard.
        </p>

        {error && (
          <div className="mb-4 p-3 bg-red-950/80 border border-red-800 rounded-xl text-red-300 text-xs">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4 text-xs">
          {isRegister && (
            <div>
              <label className="block text-slate-300 mb-1 font-medium">Full Name</label>
              <div className="relative">
                <UserIcon className="w-4 h-4 text-slate-500 absolute left-3 top-3" />
                <input
                  type="text"
                  required
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="Officer Jane Doe"
                  className="w-full pl-9 pr-3 py-2.5 bg-slate-950 border border-slate-800 rounded-xl text-slate-100 focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>
          )}

          <div>
            <label className="block text-slate-300 mb-1 font-medium">Email Address</label>
            <div className="relative">
              <Mail className="w-4 h-4 text-slate-500 absolute left-3 top-3" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="responder@disaster.org"
                className="w-full pl-9 pr-3 py-2.5 bg-slate-950 border border-slate-800 rounded-xl text-slate-100 focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>

          <div>
            <label className="block text-slate-300 mb-1 font-medium">Password</label>
            <div className="relative">
              <Lock className="w-4 h-4 text-slate-500 absolute left-3 top-3" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full pl-9 pr-3 py-2.5 bg-slate-950 border border-slate-800 rounded-xl text-slate-100 focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>

          {isRegister && (
            <div>
              <label className="block text-slate-300 mb-1 font-medium">Role Assignment</label>
              <select
                value={role}
                onChange={(e) => setRole(e.target.value as UserRole)}
                className="w-full px-3 py-2.5 bg-slate-950 border border-slate-800 rounded-xl text-slate-100 focus:outline-none focus:border-blue-500"
              >
                <option value="Citizen">Citizen</option>
                <option value="Responder">Responder</option>
                <option value="Admin">Admin</option>
              </select>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl text-xs transition-colors shadow-lg shadow-blue-600/20 disabled:opacity-50"
          >
            {loading ? 'Authenticating...' : isRegister ? 'Create Account' : 'Sign In'}
          </button>
        </form>

        {/* Quick Demo Login Credentials */}
        <div className="mt-6 pt-4 border-t border-slate-800 space-y-2">
          <span className="text-[10px] uppercase font-mono text-slate-500 block text-center">
            Quick Hackathon Demo Login
          </span>
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => handleQuickLogin('responder@disaster.org', 'Responder@123')}
              className="py-1.5 px-2 bg-slate-800 hover:bg-slate-700 text-amber-400 font-mono text-[11px] rounded-lg transition-colors border border-slate-700"
            >
              Responder Login
            </button>
            <button
              onClick={() => handleQuickLogin('admin@disaster.org', 'Admin@123')}
              className="py-1.5 px-2 bg-slate-800 hover:bg-slate-700 text-purple-400 font-mono text-[11px] rounded-lg transition-colors border border-slate-700"
            >
              Admin Login
            </button>
          </div>
        </div>

        <div className="mt-4 text-center">
          <button
            onClick={() => setIsRegister(!isRegister)}
            className="text-xs text-blue-400 hover:underline"
          >
            {isRegister ? 'Already registered? Login' : "Don't have an account? Register"}
          </button>
        </div>
      </div>
    </div>
  );
};
