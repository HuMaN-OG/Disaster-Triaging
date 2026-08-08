import React, { useState, useEffect } from 'react';
import { getIncidentBriefApi } from '../api/incidents';
import { X, FileText, Download, ShieldAlert, Cpu, Calendar, CheckSquare } from 'lucide-react';

interface ResponseBriefModalProps {
  incidentId: number;
  onClose: () => void;
}

export const ResponseBriefModal: React.FC<ResponseBriefModalProps> = ({ incidentId, onClose }) => {
  const [brief, setBrief] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getIncidentBriefApi(incidentId)
      .then((data) => setBrief(data))
      .catch((err) => {
        console.error(err);
        setError('Failed to generate response brief from backend.');
      })
      .finally(() => setLoading(false));
  }, [incidentId]);

  const downloadBrief = () => {
    if (!brief) return;
    const textContent = `DISASTER EMERGENCY RESPONSE BRIEF
=================================

[CONFIRMED DATA]
Incident ID: #${brief.incident_id}
Disaster Type: ${brief.disaster_type}
Priority Score: ${brief.priority_score.toFixed(1)} / 100
Priority Level: ${brief.priority_level}
Confidence: ${(brief.confidence * 100).toFixed(1)}%
Status: ${brief.current_status}
Language: ${brief.detected_language}
Location: ${brief.location || 'N/A'} (Lat: ${brief.latitude}, Long: ${brief.longitude})
Created At: ${new Date(brief.created_at).toLocaleString()}
Assigned Responder: ${brief.assigned_responder || 'None'}
Assigned Team: ${brief.assigned_team || 'None'}

Citizen Report:
"${brief.citizen_report || ''}"

Attachments:
${brief.attachments && brief.attachments.length > 0 ? brief.attachments.map((att: any) => ` - ${att.file_name} (${att.file_path})`).join('\n') : ' - None'}

Timeline of Events:
${brief.timeline.map((t: any) => ` - [${new Date(t.timestamp).toLocaleString()}] ${t.status}: ${t.notes || ''}`).join('\n')}

=================================
[AI/DERIVED RECOMMENDATIONS]
Risk Summary:
${brief.risk_summary}

Immediate Action Items:
${brief.immediate_actions.map((act: string) => ` - [ ] ${act}`).join('\n')}

Recommended Equipment & Resources:
${brief.recommended_resources.map((res: string) => ` - ${res}`).join('\n')}
`;
    const blob = new Blob([textContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `incident_${brief.incident_id}_response_brief.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-3xl w-full max-w-3xl max-h-[90vh] flex flex-col shadow-2xl overflow-hidden font-sans">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950/40">
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-blue-500" />
            <h2 className="text-base font-bold text-slate-100">
              Incident Response Brief #{incidentId}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 text-sm">
          {loading && (
            <div className="flex flex-col items-center justify-center py-20 space-y-3">
              <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
              <span className="text-xs text-slate-400 font-mono">Generating brief...</span>
            </div>
          )}

          {error && (
            <div className="p-4 bg-red-950/60 border border-red-800 rounded-xl text-red-300">
              {error}
            </div>
          )}

          {brief && !loading && (
            <>
              {/* Confirmed Data Section */}
              <div className="space-y-3">
                <h3 className="text-xs font-mono font-bold text-blue-400 uppercase tracking-wider flex items-center gap-1.5">
                  <ShieldAlert className="w-4 h-4 text-blue-500" /> 1. Confirmed Telemetry Data
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-slate-950/60 p-4 rounded-2xl border border-slate-800/80 font-mono text-xs">
                  <div className="space-y-1.5">
                    <div className="flex justify-between border-b border-slate-800/50 pb-1">
                      <span className="text-slate-500">Disaster Type:</span>
                      <span className="text-slate-200 font-bold">{brief.disaster_type}</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-800/50 pb-1">
                      <span className="text-slate-500">Priority Level:</span>
                      <span className={`font-bold ${brief.priority_level === 'Critical' ? 'text-red-400' : 'text-slate-200'}`}>{brief.priority_level} ({brief.priority_score.toFixed(1)})</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-800/50 pb-1">
                      <span className="text-slate-500">Confidence:</span>
                      <span className="text-slate-200">{(brief.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Current Status:</span>
                      <span className="text-slate-200">{brief.current_status}</span>
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <div className="flex justify-between border-b border-slate-800/50 pb-1">
                      <span className="text-slate-500">Language:</span>
                      <span className="text-slate-200">{brief.detected_language}</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-800/50 pb-1">
                      <span className="text-slate-500">Location:</span>
                      <span className="text-slate-200 truncate max-w-[150px]">{brief.location || 'Unknown'}</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-800/50 pb-1">
                      <span className="text-slate-500">Coordinates:</span>
                      <span className="text-slate-200">{brief.latitude.toFixed(4)}, {brief.longitude.toFixed(4)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-500">Created At:</span>
                      <span className="text-slate-200">{new Date(brief.created_at).toLocaleString()}</span>
                    </div>
                  </div>
                </div>

                {brief.citizen_report && (
                  <div className="space-y-1">
                    <span className="text-[11px] text-slate-500 font-mono">CITIZEN REPORT TEXT</span>
                    <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80 italic text-slate-300">
                      "{brief.citizen_report}"
                    </div>
                  </div>
                )}

                {brief.attachments && brief.attachments.length > 0 && (
                  <div className="space-y-1 mt-2">
                    <span className="text-[11px] text-slate-500 font-mono">ATTACHED EVIDENCE / FILES</span>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 bg-slate-950 p-3 rounded-xl border border-slate-800/80">
                      {brief.attachments.map((att: any) => (
                        <div key={att.id} className="bg-slate-900 p-2 rounded-lg border border-slate-800 flex items-center justify-between gap-2 text-xs">
                          <span className="truncate text-slate-300 font-mono text-[11px]" title={att.file_name}>
                            {att.file_name}
                          </span>
                          <a
                            href={att.file_path.startsWith('/') ? `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}${att.file_path}` : att.file_path}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-400 hover:text-blue-300 shrink-0 font-bold"
                          >
                            View
                          </a>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* AI/Derived Recommendations Section */}
              <div className="space-y-3">
                <h3 className="text-xs font-mono font-bold text-amber-500 uppercase tracking-wider flex items-center gap-1.5">
                  <Cpu className="w-4 h-4 text-amber-500" /> 2. AI & Derived Response Recommendations
                </h3>

                <div className="bg-slate-950/60 p-4 rounded-2xl border border-slate-800/80 space-y-4">
                  <div className="space-y-1">
                    <span className="text-[11px] text-slate-400 font-bold block">RISK ANALYSIS</span>
                    <p className="text-slate-200 leading-relaxed">{brief.risk_summary}</p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-1.5">
                      <span className="text-[11px] text-slate-400 font-bold block">IMMEDIATE ACTIONS</span>
                      <ul className="space-y-1 text-slate-300 list-disc list-inside">
                        {brief.immediate_actions.map((act: string, idx: number) => (
                          <li key={idx}>{act}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="space-y-1.5">
                      <span className="text-[11px] text-slate-400 font-bold block">RECOMMENDED EQUIPMENT</span>
                      <ul className="space-y-1 text-slate-300 list-disc list-inside">
                        {brief.recommended_resources.map((res: string, idx: number) => (
                          <li key={idx}>{res}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* Timeline Section */}
              {brief.timeline && brief.timeline.length > 0 && (
                <div className="space-y-3">
                  <h3 className="text-xs font-mono font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                    <Calendar className="w-4 h-4 text-slate-500" /> 3. Dispatch Timeline
                  </h3>

                  <div className="border-l border-slate-800 pl-4 space-y-3.5">
                    {brief.timeline.map((evt: any, idx: number) => (
                      <div key={idx} className="relative">
                        <div className="absolute -left-[21px] top-1.5 w-2 h-2 bg-blue-500 rounded-full" />
                        <div className="text-[11px] text-slate-500 font-mono">
                          {new Date(evt.timestamp).toLocaleString()}
                        </div>
                        <div className="text-xs font-bold text-slate-200">
                          {evt.status}
                        </div>
                        {evt.notes && <p className="text-slate-400 text-[11px] mt-0.5">{evt.notes}</p>}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer Actions */}
        <div className="px-6 py-4 border-t border-slate-800 flex items-center justify-end gap-3 bg-slate-950/40">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs font-bold transition-colors"
          >
            Close
          </button>
          {brief && (
            <button
              onClick={downloadBrief}
              className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-xs font-bold transition-colors"
            >
              <Download className="w-4 h-4" /> Download Report
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
