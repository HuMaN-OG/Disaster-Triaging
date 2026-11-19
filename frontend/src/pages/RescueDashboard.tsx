import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, AlertCircle, ArrowDownRight, ArrowRight, BarChart3, CheckCircle2, ChevronDown, Clock3, Download, Eye, Filter, Flame, LayoutList, Map as MapIcon, MapPin, RadioTower, RefreshCw, Search, ShieldAlert, SlidersHorizontal, TrendingUp, UserRoundCheck } from 'lucide-react';
import { getDashboardStatsApi, getMapLocationsApi } from '../api/dashboard';
import { getIncidentsApi } from '../api/incidents';
import { readableApiError } from '../api/client';
import { LeafletMap } from '../components/LeafletMap';
import { IncidentDrawer } from '../components/IncidentDrawer';
import type { DashboardStats, Incident, MapLocation, UserRole } from '../types';

interface RescueDashboardProps { currentUserRole: UserRole; onRequestLogin: () => void; }

function priorityClass(level: string) {
  return level === 'Critical' ? 'critical' : level === 'High' ? 'high' :
    level === 'Medium' ? 'medium' : 'low';
}
function formatDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'Unknown time' : date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}
function csvCell(v: string | number) {
  const text = String(v ?? '');
  const protectedValue = /^[=+\-@\t\r]/.test(text) ? "'" + text : text;
  return '"' + protectedValue.replace(/"/g, '""') + '"';
}
function downloadQueue(items: Incident[]) {
  const header = ['ID', 'Category', 'Priority', 'Score', 'Status', 'Created'];
  const rows = items.map(i => [i.id, i.disaster_type, i.priority_level, i.priority_score, i.status, i.created_at]);
  const csv = [header, ...rows].map(row => row.map(csvCell).join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob(['\uFEFF', csv], { type: 'text/csv;charset=utf-8' }));
  const a = document.createElement('a'); a.href = url; a.download = 'triage-incident-summary.csv'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const RescueDashboard: React.FC<RescueDashboardProps> = ({ currentUserRole, onRequestLogin }) => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [markers, setMarkers] = useState<MapLocation[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  const [search, setSearch] = useState('');
  const [priority, setPriority] = useState('all');
  const [status, setStatus] = useState('all');
  const [category, setCategory] = useState('all');
  const [context, setContext] = useState('all');
  const [sort, setSort] = useState('priority');
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [mapExpanded, setMapExpanded] = useState(false);

  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true); else setRefreshing(true);
    setError('');
    try {
      const [nextStats, nextIncidents, nextMarkers] = await Promise.all([
        getDashboardStatsApi(), getIncidentsApi({ limit: 200 }), getMapLocationsApi()
      ]);
      setStats(nextStats); setIncidents(nextIncidents); setMarkers(nextMarkers); setUpdatedAt(new Date());
    } catch (e) { setError(readableApiError(e, 'Unable to load incidents and map data.')); }
    finally { setLoading(false); setRefreshing(false); }
  }, []);

  useEffect(() => {
    void load();
    const timer = window.setInterval(() => { void load(true); }, 30000);
    return () => window.clearInterval(timer);
  }, [load]);

  const categories = useMemo(() => Array.from(new Set(incidents.map(x => x.disaster_type).filter(Boolean))).sort(), [incidents]);
  const contexts = useMemo(() => Array.from(new Set(incidents.map(x => (x as Incident & { application_context?: string }).application_context).filter((x): x is string => !!x))).sort(), [incidents]);
  const filtered = useMemo(() => incidents.filter(i => {
    if (priority !== 'all' && i.priority_level !== priority) return false;
    if (status !== 'all' && i.status !== status) return false;
    if (category !== 'all' && i.disaster_type !== category) return false;
    if (context !== 'all' && (i as Incident & { application_context?: string }).application_context !== context) return false;
    const q = search.toLowerCase().trim();
    return !q || [String(i.id), i.title, i.description || '', i.disaster_type,
      i.location?.address_text || ''].some(v => v.toLowerCase().includes(q));
  }).sort((a,b) => {
    if (sort === 'oldest') return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
    if (sort === 'newest') return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
    if (sort === 'status') return a.status.localeCompare(b.status);
    return b.priority_score - a.priority_score;
  }), [incidents, search, priority, status, category, context, sort]);

  const visibleIds = useMemo(() => new Set(filtered.map(x => x.id)), [filtered]);
  const visibleMarkers = useMemo(() => markers.filter(x => visibleIds.has(x.id)), [markers, visibleIds]);
  const canManage = currentUserRole === 'Admin' || currentUserRole === 'Responder';
  const activeCases = stats ? stats.assigned + stats.en_route + stats.in_progress : null;
  const clearFilters = () => { setSearch(''); setPriority('all'); setStatus('all'); setCategory('all'); setContext('all'); setSort('priority'); };
  const hasFilters = search || priority !== 'all' || status !== 'all' || category !== 'all' || context !== 'all';

  const statCards = [
    { name: 'Total reports', value: stats?.total_incidents, icon: RadioTower, type: 'total', note: 'In this dataset' },
    { name: 'Critical priority', value: stats?.critical, icon: Flame, type: 'critical', note: 'Requires review' },
    { name: 'Pending review', value: stats?.pending, icon: Clock3, type: 'pending', note: 'Awaiting acceptance' },
    { name: 'Active workflow', value: activeCases, icon: Activity, type: 'active', note: 'Assigned / in progress' },
    { name: 'Completed', value: stats?.completed, icon: CheckCircle2, type: 'done', note: 'Status recorded' },
  ];
  return (
    <div className="page-stack">
      <div className="command-heading">
        <div><div className="overline">OPERATIONS / INCIDENT MONITORING</div><h1>Response command<span className="heading-period">.</span></h1>
          <p>Turn incoming reports into an organized queue, map view, and human-managed response workflow.</p></div>
        <div className="command-actions">
          <span className="sync-note"><span className="sync-dot" /> {updatedAt ? 'Updated ' + updatedAt.toLocaleTimeString() : 'Waiting for API'}</span>
          <button type="button" className="btn btn-outline" disabled={refreshing || loading} onClick={() => void load(true)}><RefreshCw size={16} className={refreshing ? 'spinning' : ''} /> Refresh</button>
        </div>
      </div>

      {!canManage && <div className="access-note"><ShieldAlert size={21} />
        <div><strong>Explore the command centre in read-only mode.</strong><span>Sign in with an authorized Responder or Admin account to accept incidents and update their status.</span></div>
        <button type="button" className="btn btn-dark btn-small" onClick={onRequestLogin}>Responder sign-in <ArrowRight size={16} /></button>
      </div>}
      {error && <div className="alert-banner error" role="alert"><AlertCircle size={19}/><span>{error}</span><button type="button" className="btn btn-small btn-outline" onClick={() => void load()}>Retry</button></div>}

      <div className="command-stat-grid">
        {statCards.map(c => <div className={'command-stat '+c.type} key={c.name}><div className="stat-top"><span>{c.name}</span><c.icon size={20}/></div><div className="stat-number">{loading && !stats ? '…' : c.value ?? '—'}</div><small>{c.note}</small></div>)}
      </div>

      <div className="board-heading"><div><h2>Incident queue</h2><p>{stats ? filtered.length + ' of ' + incidents.length + ' reports shown (up to 200 latest)' : 'Data comes from the backend, not a simulation.'}</p></div>
        <div className="board-actions"><button type="button" className="btn btn-quiet" disabled={!filtered.length} onClick={() => downloadQueue(filtered)}><Download size={16}/> Export summary CSV</button>
        <button type="button" className="btn btn-outline" onClick={() => setMapExpanded(!mapExpanded)}><MapIcon size={16}/> {mapExpanded ? 'Compact map' : 'Expand map'}</button></div>
      </div>

      <div className="dt-card filter-panel">
        <div className="search-field"><Search size={18}/><input aria-label="Search incidents" value={search} onChange={e => setSearch(e.target.value)} placeholder="Search by ID, description, disaster or location" /></div>
        <div className="filter-row">
          <label><span>Priority</span><select value={priority} onChange={e => setPriority(e.target.value)}><option value="all">All priorities</option>{['Critical','High','Medium','Low','Very Low'].map(x => <option key={x}>{x}</option>)}</select></label>
          <label><span>Status</span><select value={status} onChange={e => setStatus(e.target.value)}><option value="all">All statuses</option>{['Pending','Assigned','En Route','Rescue In Progress','Completed'].map(x => <option key={x}>{x}</option>)}</select></label>
          <label><span>Category</span><select value={category} onChange={e => setCategory(e.target.value)}><option value="all">All categories</option>{categories.map(x => <option key={x}>{x}</option>)}</select></label>
          {contexts.length > 0 && <label><span>Source</span><select value={context} onChange={e=>setContext(e.target.value)}><option value="all">All sources</option>{contexts.map(x => <option key={x}>{x}</option>)}</select></label>}
          <label><span>Sort by</span><select value={sort} onChange={e => setSort(e.target.value)}><option value="priority">Highest priority</option><option value="newest">Newest first</option><option value="oldest">Oldest first</option><option value="status">Workflow status</option></select></label>
          {hasFilters && <button className="clear-filters" type="button" onClick={clearFilters}>Clear filters</button>}
        </div>
      </div>

      <div className={'command-panels'+(mapExpanded ? ' map-expanded' : '')}>
        <section className="dt-card map-panel"><div className="panel-title-row"><div><div className="overline">GEOINTELLIGENCE</div><h3>Incident map</h3></div><span className="small-count"><MapPin size={14}/>{visibleMarkers.length} mapped</span></div>
          <div className="command-map"><LeafletMap locations={visibleMarkers} center={visibleMarkers.length ? [visibleMarkers[0].latitude,visibleMarkers[0].longitude] : [20.5937,78.9629]} zoom={visibleMarkers.length ? 6 : 5} onSelectIncident={setSelectedId} /></div>
          <div className="map-legend"><span><i className="key-red"/> Critical</span><span><i className="key-orange"/> High</span><span><i className="key-yellow"/> Medium</span><span><i className="key-green"/> Low</span></div>
        </section>

        <section className="dt-card queue-panel"><div className="panel-title-row"><div><div className="overline">LIVE FROM DATABASE</div><h3>Response queue</h3></div><span className="small-count">{filtered.length} items</span></div>
          <div className="queue-table-wrap"><table className="queue-table"><thead><tr><th>Report</th><th>Priority</th><th>Status</th><th>View</th></tr></thead>
            <tbody>{filtered.length > 0 ? filtered.map(i => <tr key={i.id}>
              <td><button className="row-title-btn" type="button" onClick={() => setSelectedId(i.id)}><strong>#{i.id} · {i.disaster_type}</strong><small>{i.title || 'Citizen incident'} · {formatDate(i.created_at)}</small></button></td>
              <td><span className={'priority-pill ' + priorityClass(i.priority_level)}>{i.priority_level}</span><small className="score-mini">{i.priority_score.toFixed(1)} / 100</small></td>
              <td><span className={'status-pill ' + i.status.toLowerCase().replace(/\s+/g,'-')}>{i.status}</span></td>
              <td><button className="view-row-btn" type="button" onClick={() => setSelectedId(i.id)} aria-label={'View report ' + i.id}><ArrowRight size={17}/></button></td>
            </tr>) : <tr><td colSpan={4}><div className="empty-queue"><Filter size={24}/><strong>{loading ? 'Loading incident data…' : error ? 'Cannot load incidents' : 'Nothing in this view yet'}</strong><p>{error ? 'Check API connectivity and retry.' : hasFilters ? 'Try clearing some filters.' : 'Submitted reports appear here after the backend saves them.'}</p>{hasFilters && <button type="button" className="btn btn-outline btn-small" onClick={clearFilters}>Clear filters</button>}</div></td></tr>}</tbody>
          </table></div>
          <div className="queue-footer"><span><RefreshCw size={13}/> Auto-refresh every 30s</span><span>Click a report for details & actions <ArrowRight size={14}/></span></div>
        </section>
      </div>
      <IncidentDrawer incidentId={selectedId} userRole={currentUserRole} onClose={() => setSelectedId(null)} onIncidentUpdated={() => void load(true)} />
    </div>
  );
};
