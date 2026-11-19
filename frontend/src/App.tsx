import React, { useEffect, useState } from 'react';
import { Activity, AlertTriangle, ArrowUpRight, BarChart3, ChevronRight, CircleHelp, LayoutDashboard, LogIn, LogOut, Menu, RadioTower, ShieldCheck, Siren, Wifi, WifiOff, X } from 'lucide-react';
import { CitizenPortal } from './pages/CitizenPortal';
import { RescueDashboard } from './pages/RescueDashboard';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { LoginModal } from './components/LoginModal';
import { apiClient } from './api/client';
import { getMeApi } from './api/auth';
import type { UserRole } from './types';

type View = 'report' | 'operations' | 'analytics';
type ApiHealth = 'checking' | 'online' | 'degraded' | 'offline';
type SignedInUser = { email: string; role: UserRole; full_name: string };

const screens: { key: View; name: string; caption: string; Icon: typeof Siren }[] = [
  { key: 'report', name: 'Report an incident', caption: 'Citizen reporting', Icon: Siren },
  { key: 'operations', name: 'Response command', caption: 'Incidents & field map', Icon: LayoutDashboard },
  { key: 'analytics', name: 'Intelligence hub', caption: 'Trends & distributions', Icon: BarChart3 },
];

function App() {
  const [activeView, setActiveView] = useState<View>('report');
  const [menuOpen, setMenuOpen] = useState(false);
  const [loginOpen, setLoginOpen] = useState(false);
  const [currentUser, setCurrentUser] = useState<SignedInUser | null>(null);
  const [health, setHealth] = useState<ApiHealth>('checking');
  const [healthInfo, setHealthInfo] = useState('Checking prediction service');

  useEffect(() => {
    let live = true;
    const checkHealth = async () => {
      try {
        const response = await apiClient.get<{ status: string; database?: string; models_loaded?: boolean }>('/health', { timeout: 6000 });
        if (!live) return;
        setHealth(response.data.status === 'ok' ? 'online' : 'degraded');
        setHealthInfo(response.data.status === 'ok'
          ? 'Prediction service available'
          : 'Service partially available — check models/database');
      } catch {
        if (live) { setHealth('offline'); setHealthInfo('API not reachable — check backend setup'); }
      }
    };
    void checkHealth();
    const timer = window.setInterval(checkHealth, 45000);
    return () => { live = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => {
    if (!localStorage.getItem('disaster_triage_token')) return;
    let live = true;
    getMeApi().then(user => {
      if (live) setCurrentUser({ email: user.email, role: user.role, full_name: user.full_name });
    }).catch(() => {
      if (live) { localStorage.removeItem('disaster_triage_token'); setCurrentUser(null); }
    });
    return () => { live = false; };
  }, []);

  const setPage = (page: View) => {
    setActiveView(page);
    setMenuOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };
  const logout = () => {
    localStorage.removeItem('disaster_triage_token');
    setCurrentUser(null);
    setActiveView('report');
  };
  const section = screens.find(s => s.key === activeView)!;
  const privileged = currentUser?.role === 'Admin' || currentUser?.role === 'Responder';

  return (
    <div className="dt-app">
      {menuOpen && <button className="mobile-backdrop" aria-label="Close menu" type="button" onClick={() => setMenuOpen(false)} />}
      <aside className={'dt-sidebar' + (menuOpen ? ' is-open' : '')} aria-label="Main navigation">
        <div className="dt-brand" role="button" tabIndex={0} onClick={() => setPage('report')}
          onKeyDown={e => e.key === 'Enter' && setPage('report')}>
          <span className="brand-symbol"><RadioTower size={25} strokeWidth={2.6} /></span>
          <span className="brand-name">disaster<span>signal</span><small>INTELLIGENT TRIAGE</small></span>
        </div>

        <div className="side-instance">
          <div className="instance-top"><span className="instance-pulse" /> RESPONSE INTELLIGENCE</div>
          <strong>Field operations studio</strong>
          <span>Research prototype · 2026</span>
        </div>

        <p className="side-section-label">WORKSPACE</p>
        <nav className="side-links" aria-label="Application pages">
          {screens.map(s => (
            <button key={s.key} type="button" onClick={() => setPage(s.key)}
              className={'side-link' + (activeView === s.key ? ' active' : '')}
              aria-current={activeView === s.key ? 'page' : undefined}>
              <s.Icon size={19} strokeWidth={1.9} />
              <span>{s.name}<small>{s.caption}</small></span>
              {activeView === s.key && <ChevronRight size={16} className="side-link-arrow" />}
            </button>
          ))}
        </nav>
        <p className="side-section-label secondary-label">SYSTEM</p>
        <div className="sidebar-system">
          <div><span className={'system-light ' + health} /> <strong>AI & database</strong></div>
          <small>{healthInfo}</small>
        </div>
        <div className="sidebar-bottom">
          <div className="sidebar-footnote"><ShieldCheck size={17} /> Human oversight required</div>
          <p>Decision-support research only. Reports are not sent to emergency services.</p>
          <a href="https://github.com/HuMaN-OG/Disaster-Triaging" target="_blank" rel="noopener noreferrer">View project source <ArrowUpRight size={14} /></a>
        </div>
      </aside>

      <div className="dt-stage">
        <header className="dt-topbar">
          <div className="top-path">
            <button className="mobile-menu-btn" type="button" onClick={() => setMenuOpen(true)} aria-label="Open navigation"><Menu size={21} /></button>
            <span className="path-root">Workspace</span><ChevronRight size={15} /><strong>{section.name}</strong>
          </div>
          <div className="top-right">
            <div className={'api-chip ' + health} title={healthInfo}>
              {health === 'offline' ? <WifiOff size={14} /> : <Wifi size={14} />}
              <span>{health === 'online' ? 'API connected' : health === 'degraded' ? 'Service degraded' : health === 'offline' ? 'Backend offline' : 'Connecting…'}</span>
            </div>
            {currentUser ? (
              <div className="account-actions">
                <span className="account-avatar">{currentUser.full_name.slice(0, 1).toUpperCase()}</span>
                <span className="account-name">{currentUser.full_name}<small>{currentUser.role}</small></span>
                <button className="icon-btn" type="button" aria-label="Sign out" title="Sign out" onClick={logout}><LogOut size={19} /></button>
              </div>
            ) : (
              <button className="btn btn-dark btn-small" type="button" onClick={() => setLoginOpen(true)}>
                <LogIn size={16} /> Sign in
              </button>
            )}
          </div>
        </header>
        <main id="main-content" className="dt-content">
          <div className="prototype-notice" role="note">
            <AlertTriangle size={16} />
            <span><strong>Research / portfolio prototype.</strong> This app does not dispatch real emergency services. In an emergency, contact local emergency services immediately.</span>
            <CircleHelp size={15} className="notice-end" />
          </div>
          {activeView === 'report' && <CitizenPortal onViewOperations={() => setPage('operations')} apiAvailable={health !== 'offline'} />}
          {activeView === 'operations' && <RescueDashboard currentUserRole={currentUser?.role ?? 'Citizen'} onRequestLogin={() => setLoginOpen(true)} />}
          {activeView === 'analytics' && <AnalyticsPage />}
        </main>
        <footer className="dt-footer">
          <span><Activity size={15} /> DisasterSignal <span className="footer-muted">· Human-in-the-loop triage research</span></span>
          <span>Not an official emergency response platform · Built for portfolio demonstration</span>
        </footer>
      </div>
      <LoginModal isOpen={loginOpen} onClose={() => setLoginOpen(false)} onLoginSuccess={user => { setCurrentUser(user); setLoginOpen(false); }} />
    </div>
  );
}
export default App;
