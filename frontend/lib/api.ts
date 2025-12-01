const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_PREFIX = '/api/v1';

class ApiClient {
  private token: string | null = null;

  setToken(token: string) {
    this.token = token;
    if (typeof window !== 'undefined') {
      localStorage.setItem('token', token);
    }
  }

  getToken() {
    if (!this.token && typeof window !== 'undefined') {
      this.token = localStorage.getItem('token');
    }
    return this.token;
  }

  clearToken() {
    this.token = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('token');
    }
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const token = this.getToken();
    const headers: Record<string, string> = {
      ...(options.headers as Record<string, string>),
    };

    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }

    if (!(options.body instanceof FormData)) {
      headers['Content-Type'] = 'application/json';
    }

    const response = await fetch(`${API_BASE}${API_PREFIX}${endpoint}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  async uploadDataset(file: File, name: string, datasetType: string = 'general') {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);
    formData.append('dataset_type', datasetType);

    return this.request<any>('/datasets/upload', {
      method: 'POST',
      body: formData,
    });
  }

  async getDataset(id: number) {
    return this.request<any>(`/datasets/${id}`);
  }

  async previewDataset(id: number, rows: number = 5) {
    return this.request<any>(`/datasets/${id}/preview?rows=${rows}`);
  }

  async startAnalysis(datasetId: number, config: {
    target_column: string;
    analysis_type: string;
    feature_columns?: string;
  }) {
    const params = new URLSearchParams({
      target_column: config.target_column,
      analysis_type: config.analysis_type,
      ...(config.feature_columns && { feature_columns: config.feature_columns }),
    });

    return this.request<any>(`/analysis/auto-analyze/${datasetId}?${params}`, {
      method: 'POST',
    });
  }

  async getJobStatus(jobId: number) {
    return this.request<any>(`/jobs/${jobId}`);
  }

  async getJobResults(jobId: number) {
    return this.request<any>(`/jobs/${jobId}/results`);
  }

  async login(email: string, password: string) {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);

    const response = await this.request<any>('/auth/login', {
      method: 'POST',
      body: formData,
    });

    if (response.access_token) {
      this.setToken(response.access_token);
    }

    return response;
  }

  async register(email: string, password: string, fullName: string) {
    return this.request<any>('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password, full_name: fullName }),
    });
  }
}

export const api = new ApiClient();
