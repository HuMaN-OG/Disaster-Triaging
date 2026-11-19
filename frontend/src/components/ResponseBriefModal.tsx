import React, { useEffect, useState } from 'react';
import { AlertCircle, ArrowDownToLine, Check, ClipboardList, Clock3, FileText, MapPin, RefreshCw, ShieldCheck, X } from 'lucide-react';
import { getIncidentBriefApi } from '../api/incidents';
import { getApiAssetUrl, readableApiError } from '../api/client';

interface ResponseBriefModalProps { incidentId: number; onClose: () => void; }
interface BriefEvent { timestamp: string; status: string; notes?: string | null; }
interface BriefAttachment { id: number; file_name: string; file_path: string; }
interface IncidentBrief {
  incident_id: number; disaster_type: string; priority_score: number; priority_level: string;
  confidence: number; citizen_report: string | null; detected_language: string;
  location: string | null; latitude: number; longitude: number; created_at: string;
  current_status: string; assigned_responder: string | null; assigned_team: string | null;
  timeline: BriefEvent[]; attachments: BriefAttachment[]; risk_summary: string;
  immediate_actions: string[]; recommended_resources: string[];
}
function showDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString(undefined,{dateStyle:'medium',timeStyle:'short'});
}

export const ResponseBriefModal: React.FC<ResponseBriefModalProps> = ({ incidentId, onClose }) => {
  const [brief, setBrief] = useState<IncidentBrief | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    setLoading(true);setError('');
    getIncidentBriefApi(incidentId).then(data => {if(active)setBrief(data as IncidentBrief);})
      .catch(err => {if(active)setError(readableApiError(err,'Response brief could not be loaded.'));})
      .finally(() => {if(active)setLoading(false);});
    return () => {active=false;};
  }, [incidentId]);
  useEffect(() => {
    const close = (e: KeyboardEvent) => {if(e.key==='Escape')onClose();};
    document.addEventListener('keydown',close);
    return () => document.removeEventListener('keydown',close);
  },[onClose]);

  const download = () => {
    if (!brief) return;
    const lines = [
      'DISASTER SIGNAL — INCIDENT RESPONSE BRIEF',
      'Research prototype · Requires qualified human review',
      '',
      'CONFIRMED INCIDENT DATA',
      'Incident ID: #'+brief.incident_id,
      'Category: '+brief.disaster_type,
      'Priority: '+brief.priority_level+' ('+brief.priority_score.toFixed(1)+'/100)',
      'Model confidence: '+(brief.confidence*100).toFixed(1)+'%',
      'Status: '+brief.current_status,
      'Detected language: '+brief.detected_language,
      'Location: '+(brief.location||'Not named'),
      'Coordinates: '+brief.latitude+', '+brief.longitude,
      'Created: '+showDate(brief.created_at),
      'Assigned responder: '+(brief.assigned_responder||'Unassigned'),
      'Assigned team: '+(brief.assigned_team||'Unassigned'),
      '',
      'CITIZEN REPORT',
      brief.citizen_report||'No text report supplied',
      '',
      'AI / RULE-DERIVED ASSESSMENT (NOT VERIFIED EMERGENCY INSTRUCTIONS)',
      'Risk summary: '+brief.risk_summary,
      'Suggested review actions:',
      ...brief.immediate_actions.map(s=>'- '+s),
      'Suggested equipment:',
      ...brief.recommended_resources.map(s=>'- '+s),
      '',
      'WORKFLOW HISTORY',
      ...brief.timeline.map(x=>'- '+showDate(x.timestamp)+' — '+x.status+' '+(x.notes||'')),
      '',
      'ATTACHMENTS',
      ...brief.attachments.map(x=>'- '+x.file_name),
      '',
      'This document does not confirm that any real emergency services were dispatched.'
    ];
    const url=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/plain;charset=utf-8'}));
    const link=document.createElement('a');link.href=url;link.download='incident-'+incidentId+'-response-brief.txt';
    document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  return (
    <div className="brief-overlay" onMouseDown={e=>{if(e.target===e.currentTarget)onClose();}}>
      <section className="brief-modal" role="dialog" aria-modal="true" aria-labelledby="brief-heading">
        <header className="brief-header"><div><div className="overline">INCIDENT #{incidentId} / GENERATED DOCUMENT</div><h2 id="brief-heading">Response brief</h2><p>Confirmed records separated from model-derived suggestions.</p></div><button className="icon-button" type="button" aria-label="Close response brief" onClick={onClose}><X size={19}/></button></header>
        <div className="brief-body">
          {loading && <div className="loading-panel"><RefreshCw className="spinning" size={23}/><strong>Preparing response brief…</strong></div>}
          {error && <div className="alert-banner error" role="alert"><AlertCircle size={18}/>{error}</div>}
          {brief && !loading && <>
            <section className="brief-panel"><h3><FileText size={19}/> 01 — Confirmed case data</h3>
              <div className="brief-metrics"><div><small>Priority</small><strong>{brief.priority_level} · {brief.priority_score.toFixed(1)}/100</strong></div><div><small>Category</small><strong>{brief.disaster_type}</strong></div><div><small>Current status</small><strong>{brief.current_status}</strong></div><div><small>Confidence</small><strong>{(brief.confidence*100).toFixed(1)}%</strong></div></div>
              <div className="brief-detail-row"><MapPin size={16}/><span>{brief.location||'Coordinates recorded'} · {brief.latitude.toFixed(4)}, {brief.longitude.toFixed(4)}</span></div>
              <div className="brief-detail-row"><Clock3 size={16}/><span>{showDate(brief.created_at)}</span></div>
              <div className="brief-detail-row"><ShieldCheck size={16}/><span>Assigned: {brief.assigned_responder||'No responder'} / {brief.assigned_team||'No team'}</span></div>
              <div className="brief-report"><small>CITIZEN REPORT</small><p>{brief.citizen_report || 'Photo-only incident; no written report.'}</p></div>
              {brief.attachments?.length>0 && <div className="brief-report"><small>EVIDENCE ATTACHMENTS</small>{brief.attachments.map(a=><div className="brief-attachment" key={a.id}>{getApiAssetUrl(a.file_path)?<a href={getApiAssetUrl(a.file_path)} target="_blank" rel="noopener noreferrer">{a.file_name}</a>:<span>{a.file_name}</span>}</div>)}</div>}
            </section>
            <section className="brief-panel derived"><h3><ClipboardList size={19}/> 02 — AI-derived review recommendations</h3>
              <p className="brief-caution">Suggested actions are not verified instructions, dispatch orders, or a substitute for trained emergency personnel.</p>
              <div className="brief-subhead">RISK SUMMARY</div><p className="brief-risk">{brief.risk_summary}</p>
              <div className="brief-lists"><div><h4>Suggested review actions</h4><ul>{brief.immediate_actions.map((x,i)=><li key={i}><Check size={14}/>{x}</li>)}</ul></div>
                <div><h4>Potential resources</h4><ul>{brief.recommended_resources.map((x,i)=><li key={i}><Check size={14}/>{x}</li>)}</ul></div></div>
            </section>
            <section className="brief-panel"><h3><Clock3 size={19}/> 03 — Workflow timeline</h3>
              {brief.timeline?.length>0?<div className="brief-history">{brief.timeline.map((x,i)=><div key={i}><strong>{x.status}</strong><small>{showDate(x.timestamp)}</small>{x.notes&&<p>{x.notes}</p>}</div>)}</div>:<p className="drawer-helper">No recorded changes yet.</p>}
            </section>
          </>}
        </div>
        <footer className="brief-footer"><span>For research demonstration only.</span><div><button className="btn btn-outline btn-small" type="button" onClick={onClose}>Close</button><button className="btn btn-dark btn-small" disabled={!brief} type="button" onClick={download}><ArrowDownToLine size={16}/> Download .txt brief</button></div></footer>
      </section>
    </div>
  );
};
