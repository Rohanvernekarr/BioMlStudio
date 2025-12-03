export interface Dataset {
  id: number;
  name: string;
  file_path: string;
  file_size: number;
  dataset_type: string;
  created_at: string;
}

export interface DatasetPreview {
  dataset_id: number;
  preview_data: Record<string, any>[];
  total_rows: number;
  columns?: string[];
}

export interface Job {
  id: number;
  job_type: string;
  name: string;
  status: 'pending' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  config: any;
  progress?: number;
  result?: any;
  error_message?: string;
  created_at: string;
  updated_at?: string;
}

export interface JobResults {
  metrics?: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
    roc_auc?: number;
    mse?: number;
    rmse?: number;
    r2?: number;
  };
  feature_importance?: Array<{ feature: string; importance: number }>;
  confusion_matrix?: number[][];
  sequence_stats?: {
    total_sequences: number;
    avg_length: number;
    sequence_type: string;
  };
  plots?: {
    confusion_matrix?: string;
    feature_importance?: string;
    roc_curve?: string;
  };
  best_model?: string;
}
