import { apiClient } from './client';
import { PredictionResult } from '../types';

export const predictApi = async (formData: FormData): Promise<PredictionResult> => {
  const response = await apiClient.post<PredictionResult>('/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};
