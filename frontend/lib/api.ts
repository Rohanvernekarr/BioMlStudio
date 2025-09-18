export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";

export function authHeaders(token?: string): HeadersInit {
  const headers: HeadersInit = { };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return headers;
}

export async function apiFetch(path: string, options: RequestInit & { token?: string } = {}) {
  const { token, ...rest } = options;
  const url = `${API_BASE_URL}${path}`;
  const headers = new Headers(rest.headers || {});
  const auth = authHeaders(token);
  Object.entries(auth).forEach(([k, v]) => headers.set(k, String(v)));
  
  console.log("apiFetch:", {
    url,
    method: rest.method || 'GET',
    hasToken: !!token,
    authHeaders: auth
  });
  
  return fetch(url, { ...rest, headers });
}

// Model Builder API
export const modelBuilderAPI = {
  // Get available algorithms
  getAlgorithms: async (taskType?: string, token?: string) => {
    const params = taskType ? `?task_type=${taskType}` : '';
    const response = await apiFetch(`/model-builder/algorithms${params}`, { token });
    return response.json();
  },

  // Get algorithm suggestions
  getSuggestions: async (data: any, token?: string) => {
    const response = await apiFetch('/model-builder/suggest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
      token
    });
    return response.json();
  },

  // Validate model configuration
  validateConfig: async (config: any, token?: string) => {
    const response = await apiFetch('/model-builder/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
      token
    });
    return response.json();
  },

  // Start model training
  trainModel: async (data: any, token?: string) => {
    const response = await apiFetch('/model-builder/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
      token
    });
    return response.json();
  },

  // Start hyperparameter optimization
  optimizeHyperparameters: async (data: any, token?: string) => {
    const response = await apiFetch('/model-builder/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
      token
    });
    return response.json();
  },

  // Get preprocessors
  getPreprocessors: async (token?: string) => {
    const response = await apiFetch('/model-builder/preprocessors', { token });
    return response.json();
  }
};

// Dataset Preprocessing API
export const datasetAPI = {
  // Get preprocessing suggestions
  getPreprocessingSuggestions: async (datasetId: number, taskType: string, token?: string) => {
    const response = await apiFetch(`/datasets/${datasetId}/preprocessing-suggestions?task_type=${taskType}`, { token });
    return response.json();
  },

  // Start preprocessing
  preprocessDataset: async (datasetId: number, config: any, token?: string) => {
    const response = await apiFetch(`/datasets/${datasetId}/preprocess`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
      token
    });
    return response.json();
  },

  // Extract biological features
  extractBiologicalFeatures: async (datasetId: number, config: any, token?: string) => {
    const response = await apiFetch(`/datasets/${datasetId}/extract-biological-features`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
      token
    });
    return response.json();
  },

  // Get data quality report
  getQualityReport: async (datasetId: number, token?: string) => {
    const response = await apiFetch(`/datasets/${datasetId}/quality-report`, { token });
    return response.json();
  },

  // Upload dataset
  uploadDataset: async (formData: FormData, token?: string) => {
    console.log("API: Uploading with token:", !!token);
    const response = await apiFetch('/datasets/upload', {
      method: 'POST',
      body: formData,
      token
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error("Upload response error:", response.status, errorText);
      throw new Error(errorText);
    }
    
    return response.json();
  },

  // Get datasets
  getDatasets: async (token?: string) => {
    const response = await apiFetch('/datasets', { token });
    return response.json();
  },

  // Step 1: Data Preprocessing
  startPreprocessing: async (config: any, token?: string) => {
    const response = await fetch(`${API_BASE_URL}/jobs/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token && { Authorization: `Bearer ${token}` }),
      },
      body: JSON.stringify({
        job_type: "preprocessing",
        name: `Data Preprocessing - Dataset ${config.dataset_id}`,
        description: "Data cleaning, validation, and sequence encoding",
        config: config
      }),
    });
    
    if (!response.ok) {
      throw new Error(`Preprocessing failed: ${response.statusText}`);
    }
    
    return response.json();
  },

  // Step 2: Feature Engineering & Encoding
  startFeatureEngineering: async (config: any, token?: string) => {
    const response = await fetch(`${API_BASE_URL}/jobs/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token && { Authorization: `Bearer ${token}` }),
      },
      body: JSON.stringify({
        job_type: "feature_engineering",
        name: `Feature Engineering - Dataset ${config.dataset_id}`,
        description: "Biological feature extraction and encoding",
        config: config
      }),
    });
    
    if (!response.ok) {
      throw new Error(`Feature engineering failed: ${response.statusText}`);
    }
    
    return response.json();
  },

  // Step 3: Model Training & Selection
  startTraining: async (config: any, token?: string) => {
    const response = await fetch(`${API_BASE_URL}/jobs/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token && { Authorization: `Bearer ${token}` }),
      },
      body: JSON.stringify({
        job_type: "training",
        name: `Model Training - ${config.task_type}`,
        description: `Training ${config.task_type} models with multiple algorithms`,
        config: config
      }),
    });
    
    if (!response.ok) {
      throw new Error(`Training failed: ${response.statusText}`);
    }
    
    return response.json();
  },
};

// Jobs API
export const jobsAPI = {
  // Get job status
  getJobStatus: async (jobId: number, token?: string) => {
    const response = await apiFetch(`/jobs/${jobId}`, { token });
    return response.json();
  },

  // Get all jobs
  getJobs: async (token?: string) => {
    const response = await apiFetch('/jobs', { token });
    return response.json();
  }
};
