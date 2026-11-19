import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity, AlertCircle, ArrowUpRight, BarChart3, CalendarDays,
  CheckCircle2, Globe2, RefreshCw, ShieldCheck, TrendingUp,
} from 'lucide-react';
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Pie,
  PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { getAnalyticsApi, getDashboardStatsApi } from '../api/dashboard';
import { readableApiError } from '../api/client';
import type { AnalyticsResponse, DashboardStats } from '../types';

const palette = ['#fa6754', '#e5a04a', '#49b79a', '#587eac', '#9878ba', '#6a8c7a'];
const tip = { background: '#152d38', border: '1px solid #274352', color: '#f8fafb', borderRadius: 12, fontSize: 12 };
const entries = (values?: Record<string, number>) =>
  Object.entries(values || {}).map(([name, value]) => ({ name, value })).sort((a, b) => b.value - a.value);
const formatDay = (value: string) => {
  const dt = new Date(value + 'T12:00:00');
  return Number.isNaN(dt.getTime()) ? value : dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
};

export const AnalyticsPage: React.FC = () => {
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const refresh = useCallback(async () => {
    setLoading(true); setError('');
    try {
      const [a, s] = await Promise.all([getAnalyticsApi(), getDashboardStatsApi()]);
      setAnalytics(a); setStats(s); setLastUpdated(new Date());
    } catch (err) { setError(readableApiError(err, 'Could not retrieve analytics.')); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);

  const disaster = useMemo(() => entries(analytics?.disaster_distribution), [analytics]);
  const priorities = useMemo(() => entries(analytics?.priority_distribution), [analytics]);
  const languages = useMemo(() => entries(analytics?.language_distribution), [analytics]);
  const statuses = useMemo(() => entries(analytics?.status_distribution), [analytics]);
  const daily = useMemo(() => Object.entries(analytics?.daily_incidents || {}).sort(([a],[b]) => a.localeCompare(b))
    .map(([date, value]) => ({ date: formatDay(date), count: value })), [analytics]);
  const anyReports = !!stats?.total_incidents;

  return (
    <div className="page-stack analytics-page">
      <div className="command-heading">
        <div>
          <div className="overline">INSIGHTS / FROM THE INCIDENT DATABASE</div>
          <h1>Intelligence hub<span className="heading-period">.</span></h1>
          <p>Understand incoming report patterns and where human review may be needed.</p>
        </div>
        <div className="command-actions">
          <span className="sync-note">{lastUpdated ? 'Updated ' + lastUpdated.toLocaleTimeString() : 'Awaiting data'}</span>
          <button type="button" className="btn btn-outline" disabled={loading} onClick={() => void refresh()}>
            <RefreshCw size={16} className={loading ? 'spinning' : ''}/> Refresh insights
          </button>
        </div>
      </div>

      <div className="insights-banner">
        <div>
          <div className="overline on-dark">DECISION SUPPORT, NOT AUTOMATED DISPATCH</div>
          <h2>Find the pattern.<br/><em>Keep people in the loop.</em></h2>
          <p>Charts summarize records returned by the connected FastAPI service. Counts and confidence values are not live emergency-service statistics.</p>
        </div>
        <div className="insights-art" aria-hidden="true"><span className="bar-a" /><span className="bar-b" /><span className="bar-c" /><span className="bar-d" /><span className="bar-e" /><TrendingUp size={45}/></div>
      </div>

      {error && <div className="alert-banner error" role="alert"><AlertCircle size={19}/>{error}<button type="button" className="btn btn-outline btn-small" onClick={() => void refresh()}>Retry</button></div>}

      <div className="insight-kpis">
        <div className="insight-kpi"><span className="kpi-icon coral"><Activity size={20}/></span><div><small>Total reports</small><strong>{stats ? stats.total_incidents : '—'}</strong><p>Database incidents</p></div></div>
        <div className="insight-kpi"><span className="kpi-icon gold"><BarChart3 size={20}/></span><div><small>Vision confidence</small><strong>{analytics && anyReports ? (analytics.average_confidence*100).toFixed(1)+'%' : '—'}</strong><p>Mean recorded confidence</p></div></div>
        <div className="insight-kpi"><span className="kpi-icon teal"><CheckCircle2 size={20}/></span><div><small>Completed cases</small><strong>{stats ? stats.completed : '—'}</strong><p>Status in system</p></div></div>
        <div className="insight-kpi"><span className="kpi-icon blue"><Globe2 size={20}/></span><div><small>Languages seen</small><strong>{analytics ? languages.length : '—'}</strong><p>Detected categories</p></div></div>
      </div>

      {loading && !analytics ? <div className="loading-panel"><span className="button-spinner dark-spinner"/><strong>Reading incident analytics…</strong><p>Contacting the configured backend.</p></div> :
       !anyReports && !error ? <div className="empty-insights"><CalendarDays size={34}/><h2>Awaiting the first report</h2><p>The connected database has no incidents yet. Once a report is recorded, its distributions will appear here.</p></div> : (
        <div className="insight-charts">
          <div className="dt-card insight-chart wide">
            <div className="chart-heading"><div className="overline">REPORT ACTIVITY</div><h3>Incidents over time</h3><p>Daily number of submitted reports</p></div>
            {daily.length ? <div className="chart-body"><ResponsiveContainer width="100%" height="100%"><AreaChart data={daily} margin={{top:12,right:10,bottom:0,left:-20}}>
              <defs><linearGradient id="activityFade" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#49b79a" stopOpacity={0.35}/><stop offset="100%" stopColor="#49b79a" stopOpacity={0}/></linearGradient></defs>
              <CartesianGrid stroke="#e7ecea" strokeDasharray="4 5" vertical={false}/><XAxis dataKey="date" tick={{fontSize:11,fill:'#6b7b80'}} axisLine={false} tickLine={false}/><YAxis allowDecimals={false} tick={{fontSize:11,fill:'#6b7b80'}} axisLine={false} tickLine={false}/>
              <Tooltip contentStyle={tip}/><Area dataKey="count" type="monotone" stroke="#249c7c" strokeWidth={3} fill="url(#activityFade)" name="Reports"/>
            </AreaChart></ResponsiveContainer></div> : <p className="chart-empty">No daily data to display.</p>}
          </div>

          <div className="dt-card insight-chart">
            <div className="chart-heading"><div className="overline">CATEGORY MIX</div><h3>Disaster classification</h3><p>Number of reports by predicted class</p></div>
            {disaster.length ? <><div className="chart-body"><ResponsiveContainer width="100%" height="100%"><PieChart>
              <Pie data={disaster} dataKey="value" nameKey="name" innerRadius={63} outerRadius={96} paddingAngle={3} stroke="#fff" strokeWidth={3}>
                {disaster.map((_,i)=><Cell key={i} fill={palette[i%palette.length]}/>)}
              </Pie><Tooltip contentStyle={tip}/></PieChart></ResponsiveContainer></div>
              <div className="chart-key">{disaster.map((d,i)=><span key={d.name}><i style={{background:palette[i%palette.length]}}/>{d.name} <b>{d.value}</b></span>)}</div></> :
              <p className="chart-empty">No classification data available.</p>}
          </div>

          <div className="dt-card insight-chart">
            <div className="chart-heading"><div className="overline">TRIAGE QUEUE</div><h3>Priority distribution</h3><p>AI-estimated priority, pending human review</p></div>
            {priorities.length ? <div className="chart-body"><ResponsiveContainer width="100%" height="100%"><BarChart data={priorities} margin={{top:8,right:10,bottom:0,left:-20}}>
              <CartesianGrid stroke="#e7ecea" strokeDasharray="4 5" vertical={false}/><XAxis dataKey="name" tick={{fontSize:11,fill:'#6b7b80'}} axisLine={false} tickLine={false}/><YAxis allowDecimals={false} tick={{fontSize:11,fill:'#6b7b80'}} axisLine={false} tickLine={false}/>
              <Tooltip contentStyle={tip}/><Bar dataKey="value" name="Reports" radius={[7,7,0,0]}>{priorities.map((_,i)=><Cell key={i} fill={palette[i%palette.length]}/>)}</Bar>
            </BarChart></ResponsiveContainer></div> : <p className="chart-empty">No priority data available.</p>}
          </div>

          <div className="dt-card insight-chart">
            <div className="chart-heading"><div className="overline">ACCESSIBILITY</div><h3>Reported languages</h3><p>Detected report language mix</p></div>
            {languages.length ? <div className="chart-body"><ResponsiveContainer width="100%" height="100%"><BarChart layout="vertical" data={languages} margin={{top:8,right:12,bottom:0,left:5}}>
              <CartesianGrid stroke="#e7ecea" strokeDasharray="4 5" horizontal={false}/><XAxis type="number" allowDecimals={false} tick={{fontSize:11}} axisLine={false} tickLine={false}/>
              <YAxis type="category" dataKey="name" width={85} tick={{fontSize:11,fill:'#53686b'}} axisLine={false} tickLine={false}/><Tooltip contentStyle={tip}/>
              <Bar dataKey="value" fill="#587eac" radius={[0,7,7,0]} name="Reports"/>
            </BarChart></ResponsiveContainer></div> : <p className="chart-empty">No language data available.</p>}
          </div>

          <div className="dt-card insight-chart">
            <div className="chart-heading"><div className="overline">WORKFLOW HEALTH</div><h3>Incident status</h3><p>Where reports are in the response workflow</p></div>
            {statuses.length ? <div className="status-breakdown">{statuses.map((row,i)=>{
              const total=statuses.reduce((n,x)=>n+x.value,0);
              return <div className="status-breakdown-row" key={row.name}><div><span>{row.name}</span><strong>{row.value}</strong></div><div className="status-bar"><i style={{width: total?row.value/total*100+'%':'0%',background:palette[i%palette.length]}} /></div></div>;
            })}</div> : <p className="chart-empty">No status data available.</p>}
          </div>
        </div>
      )}
      <div className="analytics-note"><ShieldCheck size={18}/><span>These summaries are derived from a research prototype database. Model confidence is not a guarantee of correctness or an emergency-services dispatch metric.</span><ArrowUpRight size={16}/></div>
    </div>
  );
};
