import React, { useEffect, useState } from 'react';
import { AlertCircle, ArrowRight, CheckCircle2, LockKeyhole, Mail, ShieldCheck, UserRound, X } from 'lucide-react';
import { loginApi, registerApi } from '../api/auth';
import { readableApiError } from '../api/client';
import type { UserRole } from '../types';

interface LoginModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLoginSuccess: (user: { email: string; role: UserRole; full_name: string }) => void;
}

export const LoginModal: React.FC<LoginModalProps> = ({ isOpen, onClose, onLoginSuccess }) => {
  const [register, setRegister] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!isOpen) { setPassword(''); setError(''); return; }
    const escape = (e: KeyboardEvent) => { if (e.key === 'Escape' && !loading) onClose(); };
    document.addEventListener('keydown', escape);
    return () => document.removeEventListener('keydown', escape);
  }, [isOpen, onClose, loading]);

  if (!isOpen) return null;

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true); setError('');
    try {
      if (register) await registerApi({ email: email.trim(), password, full_name: fullName.trim(), role: 'Citizen' });
      const auth = await loginApi(email.trim(), password);
      localStorage.setItem('disaster_triage_token', auth.access_token);
      onLoginSuccess({ email: email.trim(), role: auth.role, full_name: auth.full_name });
      setPassword(''); onClose();
    } catch (err) { setError(readableApiError(err, 'Unable to sign in.')); }
    finally { setLoading(false); }
  };

  return (
    <div className="modal-backdrop" onMouseDown={event => { if (event.target === event.currentTarget && !loading) onClose(); }}>
      <section className="dt-login-modal" role="dialog" aria-modal="true" aria-labelledby="login-heading">
        <button type="button" className="icon-button modal-close" aria-label="Close dialog" onClick={onClose}><X size={19}/></button>
        <div className="modal-brand"><span><ShieldCheck size={25}/></span><strong>disaster<span>signal</span></strong></div>
        <div className="overline">SECURE WORKSPACE ACCESS</div>
        <h2 id="login-heading">{register ? 'Create your account.' : 'Welcome back.'}</h2>
        <p className="login-intro">{register
          ? 'Citizens can create an account to identify their research reports. Responder and Admin access must be provisioned by an administrator.'
          : 'Sign in with your configured account to manage incident status and review cases.'}</p>
        <div className="login-mode" role="group" aria-label="Authentication mode">
          <button type="button" className={!register ? 'selected' : ''} onClick={() => { setRegister(false); setError(''); }}>Sign in</button>
          <button type="button" className={register ? 'selected' : ''} onClick={() => { setRegister(true); setError(''); }}>Create citizen account</button>
        </div>
        {error && <div className="alert-banner error" role="alert"><AlertCircle size={18}/>{error}</div>}
        <form className="login-form" onSubmit={submit}>
          {register && <div><label className="field-label" htmlFor="auth-name">Full name</label><div className="icon-input"><UserRound size={18}/><input id="auth-name" type="text" value={fullName} onChange={e=>setFullName(e.target.value)} minLength={2} maxLength={100} autoComplete="name" placeholder="Your name" required /></div></div>}
          <div><label className="field-label" htmlFor="auth-email">Email address</label><div className="icon-input"><Mail size={18}/><input id="auth-email" type="email" value={email} onChange={e=>setEmail(e.target.value)} autoComplete="email" placeholder="you@example.com" required /></div></div>
          <div><label className="field-label" htmlFor="auth-password">Password</label><div className="icon-input"><LockKeyhole size={18}/><input id="auth-password" type="password" value={password} onChange={e=>setPassword(e.target.value)} autoComplete={register ? 'new-password':'current-password'} minLength={8} placeholder="Enter your password" required /></div></div>
          <button className="btn btn-coral btn-submit" disabled={loading} type="submit">
            {loading ? <><span className="button-spinner"/> {register ? 'Creating account…' : 'Signing in…'}</> : <>{register ? 'Create citizen account':'Sign in to workspace'} <ArrowRight size={18}/></>}
          </button>
        </form>
        <div className="login-security"><CheckCircle2 size={18}/><span>No publicly shared Admin/Responder passwords. Authorized roles are granted through the backend, not public registration.</span></div>
      </section>
    </div>
  );
};
