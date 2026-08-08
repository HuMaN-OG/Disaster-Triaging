import React, { useState, useEffect } from 'react';
import { getDashboardStatsApi, getMapLocationsApi } from '../api/dashboard';
import { getIncidentsApi } from '../api/incidents';
import { DashboardStats, MapLocation, Incident, UserRole } from '../types';
import { LeafletMap } from '../components/LeafletMap';
import { IncidentDrawer } from '../components/IncidentDrawer';
import {
  AlertTriangle,
  Flame,
  CheckCircle2,
  Clock,
  Shield,
  Search,
  Filter,
  RefreshCw,
  Eye,
  Activity,
  Layers,
  MapPin,
} from 'lucide-react';

interface RescueDashboardProps {
  currentUserRole?: UserRole;
}

export const RescueDashboard: React.FC<RescueDashboardProps> = ({ currentUserRole = 'Responder' }) => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [mapLocations, setMapLocations] = useState<MapLocation[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [loading, setLoading] = useState(true);

  // Filters & Search
  const [searchQuery, setSearchQuery] = useState('');
  const [priorityFilter, setPriorityFilter] = useState<string>('ALL');
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [disasterFilter, setDisasterFilter] = useState<string>('ALL');
  const [locationFilter, setLocationFilter] = useState<string>('');
  const [contextFilter, setContextFilter] = useState<string>('ALL');

  // Drawer
  const [selectedIncidentId, setSelectedIncidentId] = useState<number | null>(null);

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      const params: any = { limit: 100 };
      if (priorityFilter !== 'ALL') params.priority_level = priorityFilter;
      if (statusFilter !== 'ALL') params.status = statusFilter;
      if (disasterFilter !== 'ALL') params.disaster_type = disasterFilter;
      if (locationFilter.trim()) params.location = locationFilter;
      if (contextFilter !== 'ALL') params.application_context = contextFilter;

      const [statsData, mapData, incData] = await Promise.all([
        getDashboardStatsApi(),
        getMapLocationsApi(),
        getIncidentsApi(params),
      ]);
      setStats(statsData);
      setMapLocations(mapData);
      setIncidents(incData);
    } catch (err) {
      console.error('Failed to fetch dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    // Auto refresh telemetry every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, [priorityFilter, statusFilter, disasterFilter, locationFilter, contextFilter]);

  // Filtered Incidents
  const filteredIncidents = incidents.filter((inc) => {
    const matchesSearch =
      inc.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      inc.disaster_type.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (inc.description && inc.description.toLowerCase().includes(searchQuery.toLowerCase()));

    return matchesSearch;
  });

  return (
    <div className="max-w-7xl mx-auto px-4 lg:px-8 py-6 space-y-6">
      {/* Top Header & Refresh */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            <Activity className="w-5 h-5 text-red-500 animate-pulse" /> Rescue Operations Command Center
          </h1>
          <p className="text-xs text-slate-400 font-mono">
            Real-time disaster telemetry feeds & emergency dispatch control
          </p>
        </div>

        <button
          onClick={fetchDashboardData}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-mono text-slate-300 rounded-xl transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Telemetry</span>
        </button>
      </div>

      {/* Top Metric Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3 text-xs font-mono">
        <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-2xl">
          <span className="text-slate-500 text-[10px] block">TOTAL</span>
          <span className="text-lg font-bold text-slate-100">{stats?.total_incidents || 0}</span>
        </div>

        <div className="bg-red-950/40 border border-red-900/50 p-3 rounded-2xl">
          <span className="text-red-400 text-[10px] block">CRITICAL</span>
          <span className="text-lg font-bold text-red-400">{stats?.critical || 0}</span>
        </div>

        <div className="bg-amber-950/40 border border-amber-900/50 p-3 rounded-2xl">
          <span className="text-amber-400 text-[10px] block">HIGH</span>
          <span className="text-lg font-bold text-amber-400">{stats?.high || 0}</span>
        </div>

        <div className="bg-yellow-950/40 border border-yellow-900/50 p-3 rounded-2xl">
          <span className="text-yellow-400 text-[10px] block">MEDIUM</span>
          <span className="text-lg font-bold text-yellow-400">{stats?.medium || 0}</span>
        </div>

        <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-2xl">
          <span className="text-slate-400 text-[10px] block">LOW</span>
          <span className="text-lg font-bold text-slate-300">{(stats?.low || 0) + (stats?.very_low || 0)}</span>
        </div>

        <div className="bg-blue-950/40 border border-blue-900/50 p-3 rounded-2xl">
          <span className="text-blue-400 text-[10px] block">PENDING</span>
          <span className="text-lg font-bold text-blue-400">{stats?.pending || 0}</span>
        </div>

        <div className="bg-purple-950/40 border border-purple-900/50 p-3 rounded-2xl">
          <span className="text-purple-400 text-[10px] block">ASSIGNED</span>
          <span className="text-lg font-bold text-purple-400">{(stats?.assigned || 0) + (stats?.en_route || 0) + (stats?.in_progress || 0)}</span>
        </div>

        <div className="bg-emerald-950/40 border border-emerald-900/50 p-3 rounded-2xl">
          <span className="text-emerald-400 text-[10px] block">COMPLETED</span>
          <span className="text-lg font-bold text-emerald-400">{stats?.completed || 0}</span>
        </div>
      </div>

      {/* Main Split Layout: Interactive Map + Filterable Table */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Map */}
        <div className="lg:col-span-5 h-[500px] lg:h-auto min-h-[450px]">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-4 h-full flex flex-col space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-slate-300 font-bold flex items-center gap-1.5">
                <MapPin className="w-4 h-4 text-red-500" /> Live Disaster GIS Map
              </span>
              <span className="text-slate-500 text-[10px]">
                {mapLocations.length} Incident Markers
              </span>
            </div>
            <div className="flex-1 w-full rounded-2xl overflow-hidden border border-slate-800">
              <LeafletMap
                locations={mapLocations}
                onSelectIncident={(id) => setSelectedIncidentId(id)}
              />
            </div>
          </div>
        </div>

        {/* Right: Table */}
        <div className="lg:col-span-7 space-y-4">
          {/* Search & Filter Bar */}
          <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl flex flex-col gap-3 text-xs">
            <div className="flex flex-col sm:flex-row items-center gap-3 w-full">
              <div className="relative flex-1 w-full">
                <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search incidents by title, disaster, or text..."
                  className="w-full pl-9 pr-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-slate-100 focus:outline-none focus:border-blue-500"
                />
              </div>

              <div className="relative w-full sm:w-48">
                <input
                  type="text"
                  value={locationFilter}
                  onChange={(e) => setLocationFilter(e.target.value)}
                  placeholder="Location filter..."
                  className="w-full pl-3 pr-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-slate-100 focus:outline-none focus:border-blue-500 font-mono text-[11px]"
                />
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 w-full">
              <select
                value={priorityFilter}
                onChange={(e) => setPriorityFilter(e.target.value)}
                className="py-2 px-2.5 bg-slate-950 border border-slate-800 text-slate-300 rounded-xl focus:outline-none focus:border-blue-500 font-mono text-[11px]"
              >
                <option value="ALL">All Priorities</option>
                <option value="Critical">Critical</option>
                <option value="High">High</option>
                <option value="Medium">Medium</option>
                <option value="Low">Low</option>
              </select>

              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="py-2 px-2.5 bg-slate-950 border border-slate-800 text-slate-300 rounded-xl focus:outline-none focus:border-blue-500 font-mono text-[11px]"
              >
                <option value="ALL">All Statuses</option>
                <option value="Pending">Pending</option>
                <option value="Assigned">Assigned</option>
                <option value="En Route">En Route</option>
                <option value="Rescue In Progress">In Progress</option>
                <option value="Completed">Completed</option>
              </select>

              <select
                value={disasterFilter}
                onChange={(e) => setDisasterFilter(e.target.value)}
                className="py-2 px-2.5 bg-slate-950 border border-slate-800 text-slate-300 rounded-xl focus:outline-none focus:border-blue-500 font-mono text-[11px]"
              >
                <option value="ALL">All Categories</option>
                <option value="Flood">Flood</option>
                <option value="Earthquake">Earthquake</option>
                <option value="Urban Fire">Urban Fire</option>
                <option value="Wild Fire">Wild Fire</option>
                <option value="Landslide">Landslide</option>
                <option value="Drought">Drought</option>
              </select>

              <select
                value={contextFilter}
                onChange={(e) => setContextFilter(e.target.value)}
                className="py-2 px-2.5 bg-slate-950 border border-slate-800 text-slate-300 rounded-xl focus:outline-none focus:border-blue-500 font-mono text-[11px]"
              >
                <option value="ALL">All Contexts</option>
                <option value="Citizen Portal">Citizen Portal</option>
                <option value="Operations Center">Operations Center</option>
                <option value="API Dispatch">API Dispatch</option>
              </select>
            </div>
          </div>

          {/* Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-3xl overflow-hidden shadow-xl">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-sans">
                <thead className="bg-slate-950 text-slate-400 font-mono text-[11px] uppercase border-b border-slate-800">
                  <tr>
                    <th className="py-3 px-4">ID</th>
                    <th className="py-3 px-4">Disaster</th>
                    <th className="py-3 px-4">Priority</th>
                    <th className="py-3 px-4">Score</th>
                    <th className="py-3 px-4">Status</th>
                    <th className="py-3 px-4 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-200">
                  {filteredIncidents.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="py-12 text-center text-slate-500 font-mono">
                        No disaster incidents match the current filters.
                      </td>
                    </tr>
                  ) : (
                    filteredIncidents.map((inc) => (
                      <tr
                        key={inc.id}
                        onClick={() => setSelectedIncidentId(inc.id)}
                        className="hover:bg-slate-800/50 cursor-pointer transition-colors"
                      >
                        <td className="py-3 px-4 font-mono font-bold text-slate-400">
                          #{inc.id}
                        </td>
                        <td className="py-3 px-4 font-medium text-slate-100">
                          <span className="font-bold text-amber-400">{inc.disaster_type}</span>
                        </td>
                        <td className="py-3 px-4 font-mono">
                          <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold ${
                            inc.priority_level === 'Critical' ? 'bg-red-950 text-red-400 border border-red-800' :
                            inc.priority_level === 'High' ? 'bg-amber-950 text-amber-400 border border-amber-800' : 'bg-slate-800 text-slate-300'
                          }`}>
                            {inc.priority_level}
                          </span>
                        </td>
                        <td className="py-3 px-4 font-mono font-bold text-emerald-400">
                          {inc.priority_score.toFixed(1)}
                        </td>
                        <td className="py-3 px-4 font-mono">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                            inc.status === 'Completed' ? 'bg-emerald-950 text-emerald-400' :
                            inc.status === 'Pending' ? 'bg-blue-950 text-blue-400' : 'bg-purple-950 text-purple-400'
                          }`}>
                            {inc.status}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedIncidentId(inc.id);
                            }}
                            className="p-1.5 bg-slate-800 hover:bg-blue-600 text-slate-300 hover:text-white rounded-lg transition-colors"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Incident Detail Drawer */}
      <IncidentDrawer
        incidentId={selectedIncidentId}
        userRole={currentUserRole}
        onClose={() => setSelectedIncidentId(null)}
        onIncidentUpdated={fetchDashboardData}
      />
    </div>
  );
};
