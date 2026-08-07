import React, { useState, useEffect } from 'react';
import { getAnalyticsApi, getDashboardStatsApi } from '../api/dashboard';
import { AnalyticsResponse, DashboardStats } from '../types';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts';
import { BarChart3, PieChart as PieIcon, Globe, ShieldCheck, Cpu, RefreshCw } from 'lucide-react';

const COLORS = ['#EF4444', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6', '#EC4899'];

export const AnalyticsPage: React.FC = () => {
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchAnalytics = async () => {
    setLoading(true);
    try {
      const [anaData, statsData] = await Promise.all([
        getAnalyticsApi(),
        getDashboardStatsApi(),
      ]);
      setAnalytics(anaData);
      setStats(statsData);
    } catch (err) {
      console.error('Failed to load analytics:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, []);

  // Format data for Recharts
  const disasterChartData = analytics
    ? Object.entries(analytics.disaster_distribution).map(([name, value]) => ({ name, value }))
    : [];

  const priorityChartData = analytics
    ? Object.entries(analytics.priority_distribution).map(([name, value]) => ({ name, value }))
    : [];

  const languageChartData = analytics
    ? Object.entries(analytics.language_distribution).map(([name, value]) => ({ name, value }))
    : [];

  const statusChartData = analytics
    ? Object.entries(analytics.status_distribution).map(([name, value]) => ({ name, value }))
    : [];

  return (
    <div className="max-w-7xl mx-auto px-4 lg:px-8 py-6 space-y-8">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-blue-500" /> Triage Analytics & Model Performance Telemetry
          </h1>
          <p className="text-xs text-slate-400 font-mono">
            System performance, disaster class distribution, and language metrics
          </p>
        </div>

        <button
          onClick={fetchAnalytics}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-mono text-slate-300 rounded-xl transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Charts</span>
        </button>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl space-y-1">
          <span className="text-xs font-mono text-slate-400">Total System Incidents</span>
          <p className="text-3xl font-black font-mono text-slate-100">{stats?.total_incidents || 0}</p>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl space-y-1">
          <span className="text-xs font-mono text-slate-400">Average Vision Confidence</span>
          <p className="text-3xl font-black font-mono text-emerald-400">
            {((analytics?.average_confidence || 0) * 100).toFixed(1)}%
          </p>
        </div>

        <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl space-y-1">
          <span className="text-xs font-mono text-slate-400">Completed Rescues</span>
          <p className="text-3xl font-black font-mono text-blue-400">{stats?.completed || 0}</p>
        </div>
      </div>

      {/* Recharts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Disaster Distribution Chart */}
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-4">
          <h3 className="text-sm font-mono font-bold text-slate-200 flex items-center gap-2">
            <PieIcon className="w-4 h-4 text-amber-400" /> Disaster Class Distribution
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={disasterChartData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {disasterChartData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#fff' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Priority Level Breakdown */}
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-4">
          <h3 className="text-sm font-mono font-bold text-slate-200 flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-red-400" /> Priority Level Distribution
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={priorityChartData}>
                <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#fff' }} />
                <Bar dataKey="value" fill="#EF4444" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Language Distribution */}
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-4">
          <h3 className="text-sm font-mono font-bold text-slate-200 flex items-center gap-2">
            <Globe className="w-4 h-4 text-purple-400" /> Language Distribution (EN / HI / Hinglish)
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={languageChartData}>
                <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#fff' }} />
                <Bar dataKey="value" fill="#8B5CF6" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Rescue Status Distribution */}
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-4">
          <h3 className="text-sm font-mono font-bold text-slate-200 flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" /> Rescue Status Breakdown
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={statusChartData}>
                <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', color: '#fff' }} />
                <Bar dataKey="value" fill="#10B981" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
