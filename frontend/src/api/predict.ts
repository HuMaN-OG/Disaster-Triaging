import { apiClient } from './client';
import { PredictionResult, AuthenticityResult } from '../types';

export const predictApi = async (formData: FormData): Promise<PredictionResult> => {
  const response = await apiClient.post<PredictionResult>('/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

export const predictAuthenticityApi = async (formData: FormData): Promise<AuthenticityResult> => {
  const response = await apiClient.post<AuthenticityResult>('/predict/authenticity', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};
