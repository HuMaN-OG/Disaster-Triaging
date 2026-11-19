import axios from 'axios';

// In Docker/Nginx and Vite dev, /api is reverse-proxied to FastAPI.
// Set VITE_API_URL to a full backend URL for a separately hosted frontend.
export const API_BASE_URL: string = (import.meta.env.VITE_API_URL || '/api').replace(/\/+$/, '');

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: { Accept: 'application/json' },
});

apiClient.interceptors.request.use(config => {
  const token = localStorage.getItem('disaster_triage_token');
  if (token) config.headers.Authorization = 'Bearer ' + token;
  return config;
});

// Only resolve paths created by our own backend, never arbitrary external URLs.
export function getApiAssetUrl(path?: string | null): string {
  if (!path || !path.startsWith('/') || path.startsWith('//')) return '';
  return API_BASE_URL + path;
}

export function readableApiError(error: unknown, fallback = 'Something went wrong. Please retry.'): string {
  if (axios.isAxiosError(error)) {
    if (!error.response) return 'Could not connect to the API. Check that the backend is running and reachable.';
    const detail = error.response.data?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return detail.map(item => item.msg || 'Invalid request').join('; ');
    if (error.response.status === 403) return 'You do not have permission for this action.';
    if (error.response.status === 401) return 'Your session has expired. Please sign in again.';
    return 'Server error (' + error.response.status + '). Please retry.';
  }
  return error instanceof Error ? error.message : fallback;
}
