import { apiClient } from './client';
import type { AuthToken, User } from '../types';

export const loginApi = async (username: string, password: string): Promise<AuthToken> => {
  const response = await apiClient.post<AuthToken>('/auth/login', { username, password });
  return response.data;
};

export const registerApi = async (data: { email: string; password: string; full_name: string; role?: string }): Promise<User> => {
  const response = await apiClient.post<User>('/auth/register', data);
  return response.data;
};

export const getMeApi = async (): Promise<User> => {
  const response = await apiClient.get<User>('/auth/me');
  return response.data;
};
