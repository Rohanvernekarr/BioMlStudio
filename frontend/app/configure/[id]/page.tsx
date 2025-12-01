'use client';

import { useEffect, useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Select } from '@/components/ui/Select';
import { Header } from '@/components/Header';
import { api } from '@/lib/api';
import { useAuth } from '@/lib/useAuth';
import { Dataset, DatasetPreview } from '@/types/api';

export default function ConfigurePage() {
  useAuth();
  const router = useRouter();
  const params = useParams();
  const datasetId = Number(params.id);

  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const [analysisType, setAnalysisType] = useState('classification');
  const [targetColumn, setTargetColumn] = useState('');
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [autoPreprocess, setAutoPreprocess] = useState(true);
  const [handleImbalance, setHandleImbalance] = useState(true);
  const [normalizeFeatures, setNormalizeFeatures] = useState(true);

  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    loadData();
  }, [datasetId]);

  const loadData = async () => {
    try {
      const [datasetData, previewData] = await Promise.all([
        api.getDataset(datasetId),
        api.previewDataset(datasetId, 5),
      ]);

      setDataset(datasetData);
      setPreview(previewData);

      const columns = previewData.columns || (previewData.preview_data.length > 0 ? Object.keys(previewData.preview_data[0]) : []);
      if (columns.length > 0) {
        setSelectedFeatures(columns);
      }

      setLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dataset');
      setLoading(false);
    }
  };

  const handleTargetChange = (column: string) => {
    setTargetColumn(column);
    const columns = preview?.columns || (preview?.preview_data.length ? Object.keys(preview.preview_data[0]) : []);
    setSelectedFeatures(columns.filter((col) => col !== column));
  };

  const toggleFeature = (feature: string) => {
    setSelectedFeatures((prev) =>
      prev.includes(feature)
        ? prev.filter((f) => f !== feature)
        : [...prev, feature]
    );
  };

  const handleRunAnalysis = async () => {
    if (!targetColumn) {
      setError('Please select a target column');
      return;
    }

    setSubmitting(true);
    setError('');

    try {
      const job = await api.startAnalysis(datasetId, {
        target_column: targetColumn,
        analysis_type: analysisType,
        feature_columns: selectedFeatures.join(','),
      });

      router.push(`/running/${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start analysis');
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <>
        <Header />
        <div className="min-h-screen bg-black flex items-center justify-center">
          <div className="text-white">Loading dataset...</div>
        </div>
      </>
    );
  }

  if (!dataset || !preview) {
    return (
      <>
        <Header />
        <div className="min-h-screen bg-black flex items-center justify-center">
          <div className="text-red-400">{error || 'Dataset not found'}</div>
        </div>
      </>
    );
  }

  return (
    <>
      <Header />
      <div className="min-h-screen bg-black p-8">
        <div className="max-w-6xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">Configure Analysis</h1>
            <p className="text-zinc-400">Define your ML problem</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">"
          <div>
            <Card>
              <h2 className="text-xl font-bold mb-4">Dataset Summary</h2>
              <div className="space-y-2 text-sm mb-4">
                <p>
                  <span className="text-zinc-400">Name:</span> {dataset.name}
                </p>
                <p>
                  <span className="text-zinc-400">Samples:</span> {preview.total_rows}
                </p>
                <p>
                  <span className="text-zinc-400">Features:</span> {preview.columns?.length || (preview.preview_data.length > 0 ? Object.keys(preview.preview_data[0]).length : 0)}
                </p>
              </div>

              <h3 className="text-sm font-medium text-zinc-400 mb-2">Preview (first 5 rows)</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-zinc-800">
                      {(preview.columns || (preview.preview_data.length > 0 ? Object.keys(preview.preview_data[0]) : [])).map((col) => (
                        <th key={col} className="text-left p-2 text-zinc-400">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.preview_data.map((row, i) => {
                      const columns = preview.columns || Object.keys(row);
                      return (
                        <tr key={i} className="border-b border-zinc-800">
                          {columns.map((col) => (
                            <td key={col} className="p-2">
                              {String(row[col] ?? '')}
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>

          <div>
            <Card>
              <h2 className="text-xl font-bold mb-4">Configuration</h2>

              <div className="space-y-4">
                <Select
                  label="Analysis Type"
                  value={analysisType}
                  onChange={(e) => setAnalysisType(e.target.value)}
                >
                  <option value="classification">Classification (predict category)</option>
                  <option value="regression">Regression (predict number)</option>
                </Select>

                <Select
                  label="Target Column"
                  value={targetColumn}
                  onChange={(e) => handleTargetChange(e.target.value)}
                >
                  <option value="">Select target column</option>
                  {(preview.columns || (preview.preview_data.length > 0 ? Object.keys(preview.preview_data[0]) : [])).map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </Select>

                <div>
                  <label className="text-sm text-zinc-400 mb-2 block">
                    Input Features ({selectedFeatures.length} selected)
                  </label>
                  <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 max-h-48 overflow-y-auto">
                    {(preview.columns || (preview.preview_data.length > 0 ? Object.keys(preview.preview_data[0]) : []))
                      .filter((col) => col !== targetColumn)
                      .map((col) => (
                        <label key={col} className="flex items-center gap-2 py-1 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={selectedFeatures.includes(col)}
                            onChange={() => toggleFeature(col)}
                            className="w-4 h-4"
                          />
                          <span className="text-sm">{col}</span>
                        </label>
                      ))}
                  </div>
                </div>

                <details className="border border-zinc-800 rounded-lg">
                  <summary className="cursor-pointer p-3 text-sm font-medium">
                    Advanced Options
                  </summary>
                  <div className="p-3 pt-0 space-y-3">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={handleImbalance}
                        onChange={(e) => setHandleImbalance(e.target.checked)}
                        className="w-4 h-4"
                      />
                      <span className="text-sm">Handle class imbalance automatically</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={normalizeFeatures}
                        onChange={(e) => setNormalizeFeatures(e.target.checked)}
                        className="w-4 h-4"
                      />
                      <span className="text-sm">Normalize features</span>
                    </label>
                  </div>
                </details>
              </div>

              {error && (
                <div className="mt-4 p-3 bg-red-900/20 border border-red-800 rounded-lg text-red-400 text-sm">
                  {error}
                </div>
              )}

              <div className="mt-6 flex gap-3">
                <Button variant="secondary" onClick={() => router.push('/')}>
                  Back to Upload
                </Button>
                <Button
                  onClick={handleRunAnalysis}
                  disabled={!targetColumn || submitting}
                  className="flex-1"
                  size="lg"
                >
                  {submitting ? 'Starting...' : 'Run Analysis'}
                </Button>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
    </>
  );
}
