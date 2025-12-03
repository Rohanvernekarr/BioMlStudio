'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Header } from '@/components/Header';
import { api } from '@/lib/api';
import { useAuth } from '@/lib/useAuth';

interface AnalysisData {
  dataset_id: number;
  dataset_type: string;
  basic_stats: any;
  quality_analysis?: any;
  missing_data?: any;
  recommendations?: string[];
  column_info?: any;
}

interface VisualizationData {
  dataset_id: number;
  plots: { [key: string]: string };
  plot_descriptions?: { [key: string]: string };
}

export default function AnalysisPage() {
  useAuth();
  const params = useParams();
  const router = useRouter();
  const datasetId = parseInt(params.id as string);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [dataset, setDataset] = useState<any>(null);
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [visualizations, setVisualizations] = useState<VisualizationData | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'quality' | 'visualizations'>('overview');

  useEffect(() => {
    loadAnalysisData();
  }, [datasetId]);

  const loadAnalysisData = async () => {
    setLoading(true);
    setError('');

    try {
      // Load dataset info
      const datasetInfo = await api.getDataset(datasetId);
      setDataset(datasetInfo);

      // Load analysis
      const analysisData = await api.analyzeDataset(datasetId);
      setAnalysis(analysisData);

      // Load visualizations
      const vizData = await api.visualizeDataset(datasetId);
      setVisualizations(vizData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load analysis');
    } finally {
      setLoading(false);
    }
  };

  const renderBasicStats = () => {
    if (!analysis?.basic_stats) return null;

    const stats = analysis.basic_stats;

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        {stats.total_rows && (
          <Card>
            <h3 className="text-sm font-medium text-zinc-400 mb-1">Total Rows</h3>
            <p className="text-2xl font-bold">{stats.total_rows.toLocaleString()}</p>
          </Card>
        )}
        {stats.total_columns && (
          <Card>
            <h3 className="text-sm font-medium text-zinc-400 mb-1">Total Columns</h3>
            <p className="text-2xl font-bold">{stats.total_columns}</p>
          </Card>
        )}
        {stats.sequence_count && (
          <Card>
            <h3 className="text-sm font-medium text-zinc-400 mb-1">Sequences</h3>
            <p className="text-2xl font-bold">{stats.sequence_count.toLocaleString()}</p>
          </Card>
        )}
        {stats.avg_sequence_length && (
          <Card>
            <h3 className="text-sm font-medium text-zinc-400 mb-1">Avg Length</h3>
            <p className="text-2xl font-bold">{Math.round(stats.avg_sequence_length)}</p>
          </Card>
        )}
        {stats.min_sequence_length && (
          <Card>
            <h3 className="text-sm font-medium text-zinc-400 mb-1">Min Length</h3>
            <p className="text-2xl font-bold">{stats.min_sequence_length}</p>
          </Card>
        )}
        {stats.max_sequence_length && (
          <Card>
            <h3 className="text-sm font-medium text-zinc-400 mb-1">Max Length</h3>
            <p className="text-2xl font-bold">{stats.max_sequence_length}</p>
          </Card>
        )}
        {stats.mean && (
          <Card>
            <h3 className="text-sm font-medium text-zinc-400 mb-1">GC Content</h3>
            <p className="text-2xl font-bold">{(stats.mean * 100).toFixed(2)}%</p>
          </Card>
        )}
      </div>
    );
  };

  const renderQualityAnalysis = () => {
    if (!analysis?.quality_analysis) {
      return (
        <Card>
          <p className="text-zinc-400">No quality analysis available for this dataset type.</p>
        </Card>
      );
    }

    const qa = analysis.quality_analysis;

    return (
      <div className="space-y-6">
        <Card>
          <h3 className="text-lg font-semibold mb-4">Quality Summary</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-zinc-400">Total Sequences</p>
              <p className="text-2xl font-bold">{qa.total_sequences}</p>
            </div>
            <div>
              <p className="text-sm text-zinc-400">Sequences with Issues</p>
              <p className="text-2xl font-bold text-yellow-500">{qa.sequences_with_issues}</p>
            </div>
          </div>
        </Card>

        {qa.ambiguous_bases && (
          <Card>
            <h3 className="text-lg font-semibold mb-4">Ambiguous Bases</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-zinc-400">Total Count</p>
                <p className="text-xl font-bold">{qa.ambiguous_bases.total_count}</p>
              </div>
              <div>
                <p className="text-sm text-zinc-400">Sequences Affected</p>
                <p className="text-xl font-bold">{qa.ambiguous_bases.sequences_affected}</p>
              </div>
              <div>
                <p className="text-sm text-zinc-400">Percentage</p>
                <p className="text-xl font-bold">{qa.ambiguous_bases.percentage.toFixed(2)}%</p>
              </div>
            </div>
          </Card>
        )}

        {qa.gaps && (
          <Card>
            <h3 className="text-lg font-semibold mb-4">Gaps</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-zinc-400">Total Count</p>
                <p className="text-xl font-bold">{qa.gaps.total_count}</p>
              </div>
              <div>
                <p className="text-sm text-zinc-400">Sequences Affected</p>
                <p className="text-xl font-bold">{qa.gaps.sequences_affected}</p>
              </div>
              <div>
                <p className="text-sm text-zinc-400">Percentage</p>
                <p className="text-xl font-bold">{qa.gaps.percentage.toFixed(2)}%</p>
              </div>
            </div>
          </Card>
        )}

        {qa.issues && qa.issues.length > 0 && (
          <Card>
            <h3 className="text-lg font-semibold mb-4">Detected Issues</h3>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {qa.issues.map((issue: any, idx: number) => (
                <div key={idx} className="p-3 bg-zinc-800 rounded border border-zinc-700">
                  <p className="text-sm font-medium text-zinc-300">
                    Sequence {issue.sequence_index + 1}
                  </p>
                  <ul className="mt-2 space-y-1">
                    {issue.problems.map((problem: string, pIdx: number) => (
                      <li key={pIdx} className="text-sm text-zinc-400 ml-4 list-disc">
                        {problem}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>
    );
  };

  const renderMissingData = () => {
    if (!analysis?.missing_data && !analysis?.column_info?.missing_values) {
      return null;
    }

    const missingData = analysis.missing_data;
    const missingValues = analysis.column_info?.missing_values;

    return (
      <Card>
        <h3 className="text-lg font-semibold mb-4">Missing Data Analysis</h3>
        
        {missingData && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {missingData.empty_sequences > 0 && (
              <div>
                <p className="text-sm text-zinc-400">Empty Sequences</p>
                <p className="text-xl font-bold text-red-500">{missingData.empty_sequences}</p>
              </div>
            )}
            {missingData.sequences_with_all_N > 0 && (
              <div>
                <p className="text-sm text-zinc-400">All N Sequences</p>
                <p className="text-xl font-bold text-red-500">{missingData.sequences_with_all_N}</p>
              </div>
            )}
            {missingData.sequences_mostly_gaps > 0 && (
              <div>
                <p className="text-sm text-zinc-400">Mostly Gaps</p>
                <p className="text-xl font-bold text-yellow-500">{missingData.sequences_mostly_gaps}</p>
              </div>
            )}
          </div>
        )}

        {missingValues && Object.keys(missingValues).length > 0 && (
          <div>
            <h4 className="text-md font-medium mb-3">Missing Values by Column</h4>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {Object.entries(missingValues).map(([col, info]: [string, any]) => (
                <div key={col} className="flex items-center justify-between p-2 bg-zinc-800 rounded">
                  <span className="text-sm font-medium">{col}</span>
                  <div className="flex items-center gap-4">
                    <span className="text-sm text-zinc-400">{info.count} missing</span>
                    <span className="text-sm font-bold text-yellow-500">
                      {info.percentage.toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </Card>
    );
  };

  const renderRecommendations = () => {
    if (!analysis?.recommendations || analysis.recommendations.length === 0) {
      return null;
    }

    return (
      <Card>
        <h3 className="text-lg font-semibold mb-4">Recommendations</h3>
        <ul className="space-y-2">
          {analysis.recommendations.map((rec, idx) => (
            <li key={idx} className="flex items-start gap-3">
              <span className="text-blue-500 mt-1">•</span>
              <span className="text-sm text-zinc-300">{rec}</span>
            </li>
          ))}
        </ul>
      </Card>
    );
  };

  const renderVisualizations = () => {
    if (!visualizations?.plots) {
      return (
        <Card>
          <p className="text-zinc-400">No visualizations available.</p>
        </Card>
      );
    }

    return (
      <div className="space-y-6">
        {Object.entries(visualizations.plots).map(([plotName, plotData]) => (
          <Card key={plotName}>
            <h3 className="text-lg font-semibold mb-4 capitalize">
              {plotName.replace(/_/g, ' ')}
            </h3>
            {visualizations.plot_descriptions?.[plotName] && (
              <p className="text-sm text-zinc-400 mb-4">
                {visualizations.plot_descriptions[plotName]}
              </p>
            )}
            <div className="bg-white rounded-lg p-4">
              <img
                src={`data:image/png;base64,${plotData}`}
                alt={plotName}
                className="w-full h-auto"
              />
            </div>
          </Card>
        ))}
      </div>
    );
  };

  if (loading) {
    return (
      <>
        <Header />
        <div className="min-h-screen bg-black flex items-center justify-center">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-white mb-4"></div>
            <p className="text-zinc-400">Analyzing dataset...</p>
          </div>
        </div>
      </>
    );
  }

  if (error) {
    return (
      <>
        <Header />
        <div className="min-h-screen bg-black flex items-center justify-center p-8">
          <Card className="max-w-md w-full">
            <h2 className="text-xl font-bold text-red-500 mb-4">Error</h2>
            <p className="text-zinc-400 mb-6">{error}</p>
            <Button onClick={() => router.push('/')}>
              Back to Home
            </Button>
          </Card>
        </div>
      </>
    );
  }

  return (
    <>
      <Header />
      <div className="min-h-screen bg-black py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-12">
            <Button
              variant="ghost"
              onClick={() => router.back()}
              className="mb-4"
            >
              ← Back
            </Button>
            <h1 className="text-4xl sm:text-5xl font-bold mb-4">Dataset Analysis</h1>
            {dataset && (
              <div className="flex items-center gap-4 text-zinc-400">
                <span>{dataset.name}</span>
                <span>•</span>
                <span className="capitalize">{dataset.dataset_type}</span>
                <span>•</span>
                <span>{(dataset.file_size / 1024).toFixed(2)} KB</span>
              </div>
            )}
          </div>

          {/* Tabs */}
          <div className="flex gap-4 mb-6 border-b border-zinc-800">
            <button
              className={`pb-3 px-4 font-medium transition-colors ${
                activeTab === 'overview'
                  ? 'text-white border-b-2 border-white'
                  : 'text-zinc-400 hover:text-zinc-300'
              }`}
              onClick={() => setActiveTab('overview')}
            >
              Overview
            </button>
            <button
              className={`pb-3 px-4 font-medium transition-colors ${
                activeTab === 'quality'
                  ? 'text-white border-b-2 border-white'
                  : 'text-zinc-400 hover:text-zinc-300'
              }`}
              onClick={() => setActiveTab('quality')}
            >
              Quality Analysis
            </button>
            <button
              className={`pb-3 px-4 font-medium transition-colors ${
                activeTab === 'visualizations'
                  ? 'text-white border-b-2 border-white'
                  : 'text-zinc-400 hover:text-zinc-300'
              }`}
              onClick={() => setActiveTab('visualizations')}
            >
              Visualizations
            </button>
          </div>

          {/* Tab Content */}
          {activeTab === 'overview' && (
            <div className="space-y-6">
              <h2 className="text-2xl font-semibold">Basic Statistics</h2>
              {renderBasicStats()}
              {renderMissingData()}
              {renderRecommendations()}
            </div>
          )}

          {activeTab === 'quality' && (
            <div className="space-y-6">
              <h2 className="text-2xl font-semibold">Quality Metrics</h2>
              {renderQualityAnalysis()}
            </div>
          )}

          {activeTab === 'visualizations' && (
            <div className="space-y-6">
              <h2 className="text-2xl font-semibold">Data Visualizations</h2>
              {renderVisualizations()}
            </div>
          )}

          {/* Action Buttons */}
          <div className="mt-8 flex gap-4">
            <Button onClick={() => router.push(`/configure/${datasetId}`)}>
              Continue to Model Configuration
            </Button>
            <Button variant="outline" onClick={loadAnalysisData}>
              Refresh Analysis
            </Button>
          </div>
        </div>
      </div>
    </>
  );
}
