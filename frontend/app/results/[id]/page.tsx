'use client';

import { useEffect, useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { MetricCard } from '@/components/ui/MetricCard';
import { Header } from '@/components/Header';
import { api } from '@/lib/api';
import { useAuth } from '@/lib/useAuth';
import { Job, JobResults } from '@/types/api';

export default function ResultsPage() {
  useAuth();
  const router = useRouter();
  const params = useParams();
  const jobId = Number(params.id);

  const [job, setJob] = useState<Job | null>(null);
  const [results, setResults] = useState<JobResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadResults();
  }, [jobId]);

  const loadResults = async () => {
    try {
      const resultsData = await api.getWorkflowResults(jobId);
      setJob(resultsData as any);
      setResults(resultsData.results as any);
      setLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load results');
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <>
        <Header />
        <div className="min-h-screen bg-black flex items-center justify-center">
          <div className="text-white">Loading results...</div>
        </div>
      </>
    );
  }

  if (error || !results) {
    return (
      <>
        <Header />
        <div className="min-h-screen bg-black flex items-center justify-center">
          <div className="text-red-400">{error || 'Results not found'}</div>
        </div>
      </>
    );
  }

  const metrics = results.metrics || {};
  const featureImportanceObj = results.feature_importance || {};
  const featureImportance = Object.entries(featureImportanceObj).map(([feature, importance]) => ({
    feature,
    importance: importance as number
  })).sort((a, b) => b.importance - a.importance);
  const confusionMatrix = results.confusion_matrix || [];
  const sequenceStats = results.sequence_stats || null;

  const isClassification = metrics.accuracy !== undefined;
  const maxImportance = featureImportance.length > 0 ? Math.max(...featureImportance.map((f) => f.importance)) : 1;

  return (
    <>
      <Header />
      <div className="min-h-screen bg-black py-16 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-16 flex flex-col sm:flex-row justify-between items-start gap-6">
            <div>
              <h1 className="text-4xl sm:text-5xl font-bold mb-4 tracking-tight">Training Results</h1>
              <p className="text-zinc-400 text-lg">
                {job?.name && <span className="text-zinc-500">{job.name} • </span>}
                Best model: <span className="text-white font-medium">{results.best_model || 'RandomForest'}</span>
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <Button
                variant="secondary"
                onClick={() => window.open(api.getWorkflowModelDownloadUrl(jobId))}
                size="lg"
              >
                📦 Download Model
              </Button>
              <Button
                onClick={() => window.open(api.getWorkflowReportDownloadUrl(jobId), '_blank')}
                size="lg"
              >
                📄 Download Report
              </Button>
            </div>
          </div>

        {sequenceStats && (
          <Card className="mb-8">
            <h2 className="text-2xl font-bold mb-6">Sequence Data Summary</h2>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
              <div className="p-4 rounded-lg bg-zinc-950/50 border border-zinc-800/50">
                <p className="text-sm text-zinc-400 mb-2">Total Sequences</p>
                <p className="text-3xl font-bold">{sequenceStats.total_sequences}</p>
              </div>
              <div className="p-4 rounded-lg bg-zinc-950/50 border border-zinc-800/50">
                <p className="text-sm text-zinc-400 mb-2">Average Length</p>
                <p className="text-3xl font-bold">{sequenceStats.avg_length?.toFixed(0) || 0} <span className="text-lg text-zinc-500">bp</span></p>
              </div>
              <div className="p-4 rounded-lg bg-zinc-950/50 border border-zinc-800/50">
                <p className="text-sm text-zinc-400 mb-2">Sequence Type</p>
                <p className="text-3xl font-bold capitalize">{sequenceStats.sequence_type}</p>
              </div>
            </div>
          </Card>
        )}

        {results.models_trained && results.models_trained.length > 0 && (
          <Card className="mb-8">
            <h2 className="text-2xl font-bold mb-4">Model Comparison</h2>
            <p className="text-sm text-zinc-400 mb-6">
              Performance comparison of all trained models • Total training time: {results.training_time?.toFixed(2) || 0}s
            </p>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-zinc-800">
                    <th className="text-left py-3 px-4 text-sm font-semibold text-zinc-400">Model</th>
                    <th className="text-left py-3 px-4 text-sm font-semibold text-zinc-400">Type</th>
                    <th className="text-right py-3 px-4 text-sm font-semibold text-zinc-400">Train Score</th>
                    <th className="text-right py-3 px-4 text-sm font-semibold text-zinc-400">Val Score</th>
                    <th className="text-right py-3 px-4 text-sm font-semibold text-zinc-400">Training Time</th>
                    <th className="text-center py-3 px-4 text-sm font-semibold text-zinc-400">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {results.models_trained
                    .sort((a, b) => (b.metrics?.primary_score || 0) - (a.metrics?.primary_score || 0))
                    .map((model, idx) => (
                    <tr 
                      key={idx} 
                      className={`border-b border-zinc-800/50 transition-colors ${
                        model.is_best ? 'bg-white/5' : 'hover:bg-zinc-900/50'
                      }`}
                    >
                      <td className="py-4 px-4">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{model.model_name}</span>
                          {model.is_best && (
                            <span className="px-2 py-1 text-xs font-semibold bg-green-500/20 text-green-400 rounded-full">
                              BEST
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="py-4 px-4 text-zinc-400 text-sm">{model.model_type}</td>
                      <td className="py-4 px-4 text-right font-mono">
                        {model.metrics?.train_score?.toFixed(4) || 'N/A'}
                      </td>
                      <td className="py-4 px-4 text-right font-mono">
                        <span className={model.metrics?.val_score && model.metrics.val_score > 0.85 ? 'text-green-400' : ''}>
                          {model.metrics?.val_score?.toFixed(4) || 'N/A'}
                        </span>
                      </td>
                      <td className="py-4 px-4 text-right text-zinc-400 font-mono text-sm">
                        {model.training_time?.toFixed(2) || 0}s
                      </td>
                      <td className="py-4 px-4 text-center">
                        {idx === 0 ? (
                          <span className="text-yellow-400">🥇</span>
                        ) : idx === 1 ? (
                          <span className="text-zinc-400">🥈</span>
                        ) : idx === 2 ? (
                          <span className="text-orange-400">🥉</span>
                        ) : (
                          <span className="text-zinc-600">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-6 p-4 rounded-lg bg-zinc-900/50 border border-zinc-800">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
                <div>
                  <p className="text-zinc-500 mb-1">Best Model</p>
                  <p className="font-semibold text-white">{results.best_model}</p>
                </div>
                <div>
                  <p className="text-zinc-500 mb-1">Models Trained</p>
                  <p className="font-semibold text-white">{results.models_trained.length}</p>
                </div>
                <div>
                  <p className="text-zinc-500 mb-1">Fastest Model</p>
                  <p className="font-semibold text-white">
                    {results.models_trained.reduce((fastest, model) => 
                      (model.training_time < fastest.training_time) ? model : fastest
                    ).model_name}
                  </p>
                </div>
              </div>
            </div>
          </Card>
        )}

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">{isClassification ? (
            <>
              <MetricCard
                label="Accuracy"
                value={metrics.accuracy || 0}
                good={(metrics.accuracy || 0) > 0.85}
              />
              <MetricCard
                label="F1 Score"
                value={metrics.f1_score || 0}
                good={(metrics.f1_score || 0) > 0.8}
              />
              <MetricCard
                label="Precision"
                value={metrics.precision || 0}
                good={(metrics.precision || 0) > 0.8}
              />
              <MetricCard
                label="Recall"
                value={metrics.recall || 0}
                good={(metrics.recall || 0) > 0.8}
              />
            </>
          ) : (
            <>
              <MetricCard label="R² Score" value={metrics.r2 || 0} good={(metrics.r2 || 0) > 0.8} />
              <MetricCard label="MSE" value={metrics.mse || 0} />
              <MetricCard label="RMSE" value={metrics.rmse || 0} />
            </>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <Card>
            <h2 className="text-2xl font-bold mb-2">Top Features</h2>
            <p className="text-sm text-zinc-400 mb-6">
              Features that most influence the model's predictions
            </p>
            <div className="space-y-4">
              {featureImportance.slice(0, 10).map((item, i) => (
                <div key={i}>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="font-medium">{item.feature}</span>
                    <span className="text-zinc-400 font-mono">{item.importance.toFixed(3)}</span>
                  </div>
                  <div className="w-full bg-zinc-800 rounded-full h-2.5">
                    <div
                      className="bg-white h-2.5 rounded-full transition-all"
                      style={{ width: `${(item.importance / maxImportance) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {isClassification && confusionMatrix.length > 0 && (
            <Card>
              <h2 className="text-2xl font-bold mb-2">Confusion Matrix</h2>
              <p className="text-sm text-zinc-400 mb-6">
                Actual vs predicted classifications
              </p>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <tbody>
                    {confusionMatrix.map((row, i) => (
                      <tr key={i}>
                        {row.map((cell, j) => {
                          const max = Math.max(...confusionMatrix.flat());
                          const intensity = cell / max;
                          return (
                            <td
                              key={j}
                              className="p-6 text-center border border-zinc-800 font-medium transition-all hover:scale-105"
                              style={{
                                backgroundColor: `rgba(255, 255, 255, ${intensity * 0.3})`,
                              }}
                            >
                              {cell}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>

        <Card className="mb-6">
          <h2 className="text-xl font-bold mb-4">Summary</h2>
          <ul className="space-y-3 text-zinc-300 list-disc list-inside">
            {isClassification ? (
              <>
                <li>
                  The model correctly classified {((metrics.accuracy || 0) * 100).toFixed(1)}% of
                  samples with {((metrics.precision || 0) * 100).toFixed(1)}% precision.
                </li>
                <li>
                  Top contributing features: {featureImportance.slice(0, 3).map((f) => f.feature).join(', ')}.
                </li>
                {sequenceStats && (
                  <li>
                    Analyzed {sequenceStats.total_sequences} {sequenceStats.sequence_type} sequences
                    with average length of {sequenceStats.avg_length.toFixed(0)} base pairs.
                  </li>
                )}
              </>
            ) : (
              <>
                <li>
                  The model achieved an R² score of {(metrics.r2 || 0).toFixed(3)}, explaining{' '}
                  {((metrics.r2 || 0) * 100).toFixed(1)}% of variance.
                </li>
                <li>
                  Top contributing features: {featureImportance.slice(0, 3).map((f) => f.feature).join(', ')}.
                </li>
              </>
            )}
          </ul>
        </Card>

        {sequenceStats && featureImportance.some(f => f.feature.startsWith('kmer_')) && (
          <Card className="mb-6">
            <h2 className="text-xl font-bold mb-4">Biological Insights</h2>
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-semibold mb-2 text-zinc-200">Key Sequence Patterns</h3>
                <p className="text-zinc-400 mb-3">
                  K-mer analysis identified the following sequence motifs as most predictive:
                </p>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {featureImportance
                    .filter(f => f.feature.startsWith('kmer_'))
                    .slice(0, 6)
                    .map((f, i) => (
                      <div key={i} className="bg-zinc-900 p-3 rounded-lg border border-zinc-800">
                        <div className="font-mono text-white font-bold text-lg mb-1">
                          {f.feature.replace('kmer_', '')}
                        </div>
                        <div className="text-sm text-zinc-400">
                          Importance: {(f.importance * 100).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                </div>
              </div>
              
              <div className="pt-4 border-t border-zinc-800">
                <h3 className="text-lg font-semibold mb-2 text-zinc-200">Interpretation</h3>
                <p className="text-zinc-400">
                  These sequence patterns show strong correlation with the target variable, 
                  suggesting potential functional or structural significance. 
                  {sequenceStats.sequence_type === 'dna' && ' These DNA motifs may represent regulatory elements, binding sites, or conserved regions.'}
                  {sequenceStats.sequence_type === 'protein' && ' These protein patterns may indicate functional domains or structural motifs.'}
                </p>
              </div>
            </div>
          </Card>
        )}

        <div className="mt-8 flex gap-4">
          <Button onClick={() => router.push('/')} size="lg">
            Run New Analysis
          </Button>
          <Button
            onClick={() => {
              const link = document.createElement('a');
              link.href = `${process.env.NEXT_PUBLIC_API_URL}/analysis/download-model/${jobId}`;
              link.download = `model_${jobId}.joblib`;
              const token = localStorage.getItem('token');
              if (token) {
                fetch(link.href, {
                  headers: { 'Authorization': `Bearer ${token}` }
                }).then(res => res.blob()).then(blob => {
                  const url = window.URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `model_${jobId}.joblib`;
                  a.click();
                });
              }
            }}
            variant="secondary"
            size="lg"
          >
            Download Model
          </Button>
          <Button
            onClick={() => {
              const reportContent = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>BioMLStudio Analysis Report - Job ${jobId}</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 1200px; margin: 40px auto; padding: 20px; background: #000; color: #fff; }
    h1 { color: #fff; border-bottom: 2px solid #27272a; padding-bottom: 10px; }
    h2 { color: #e4e4e7; margin-top: 30px; }
    .metric { display: inline-block; background: #18181b; padding: 15px 25px; margin: 10px; border-radius: 8px; border: 1px solid #27272a; }
    .metric-label { color: #a1a1aa; font-size: 14px; }
    .metric-value { font-size: 32px; font-weight: bold; color: #fff; }
    .feature-bar { background: #27272a; height: 8px; border-radius: 4px; margin: 10px 0; }
    .feature-fill { background: #fff; height: 100%; border-radius: 4px; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    td { padding: 12px; text-align: center; border: 1px solid #27272a; background: #18181b; }
    .kmer-box { display: inline-block; background: #18181b; padding: 12px 20px; margin: 8px; border-radius: 8px; border: 1px solid #27272a; }
    .kmer-seq { font-family: monospace; font-size: 18px; font-weight: bold; }
    .summary { background: #18181b; padding: 20px; border-radius: 8px; border: 1px solid #27272a; margin: 20px 0; }
  </style>
</head>
<body>
  <h1>BioMLStudio Analysis Report</h1>
  <p><strong>Job:</strong> ${job?.name || 'Analysis'} • <strong>Model:</strong> ${results.best_model || 'RandomForest'}</p>
  <p><strong>Date:</strong> ${new Date().toLocaleDateString()}</p>
  
  ${sequenceStats ? `
  <h2>Sequence Data Summary</h2>
  <div class="metric">
    <div class="metric-label">Total Sequences</div>
    <div class="metric-value">${sequenceStats.total_sequences}</div>
  </div>
  <div class="metric">
    <div class="metric-label">Average Length</div>
    <div class="metric-value">${sequenceStats.avg_length.toFixed(0)} bp</div>
  </div>
  <div class="metric">
    <div class="metric-label">Sequence Type</div>
    <div class="metric-value" style="text-transform: capitalize">${sequenceStats.sequence_type}</div>
  </div>
  ` : ''}
  
  <h2>Performance Metrics</h2>
  ${isClassification ? `
  <div class="metric">
    <div class="metric-label">Accuracy</div>
    <div class="metric-value">${((metrics.accuracy || 0) * 100).toFixed(1)}%</div>
  </div>
  <div class="metric">
    <div class="metric-label">Precision</div>
    <div class="metric-value">${((metrics.precision || 0) * 100).toFixed(1)}%</div>
  </div>
  <div class="metric">
    <div class="metric-label">Recall</div>
    <div class="metric-value">${((metrics.recall || 0) * 100).toFixed(1)}%</div>
  </div>
  <div class="metric">
    <div class="metric-label">F1 Score</div>
    <div class="metric-value">${((metrics.f1_score || 0) * 100).toFixed(1)}%</div>
  </div>
  ` : `
  <div class="metric">
    <div class="metric-label">R² Score</div>
    <div class="metric-value">${(metrics.r2 || 0).toFixed(3)}</div>
  </div>
  <div class="metric">
    <div class="metric-label">MSE</div>
    <div class="metric-value">${(metrics.mse || 0).toFixed(3)}</div>
  </div>
  `}
  
  <h2>Top Features</h2>
  ${featureImportance.slice(0, 10).map(f => `
    <div style="margin: 15px 0">
      <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
        <span>${f.feature}</span>
        <span style="color: #a1a1aa">${(f.importance * 100).toFixed(1)}%</span>
      </div>
      <div class="feature-bar">
        <div class="feature-fill" style="width: ${(f.importance / maxImportance) * 100}%"></div>
      </div>
    </div>
  `).join('')}
  
  ${sequenceStats && featureImportance.some(f => f.feature.startsWith('kmer_')) ? `
  <h2>Key Sequence Patterns</h2>
  <div>
    ${featureImportance.filter(f => f.feature.startsWith('kmer_')).slice(0, 6).map(f => `
      <div class="kmer-box">
        <div class="kmer-seq">${f.feature.replace('kmer_', '')}</div>
        <div style="color: #a1a1aa; font-size: 12px; margin-top: 5px;">Importance: ${(f.importance * 100).toFixed(1)}%</div>
      </div>
    `).join('')}
  </div>
  ` : ''}
  
  <div class="summary">
    <h2 style="margin-top: 0">Summary</h2>
    ${isClassification ? `
      <p>The model correctly classified ${((metrics.accuracy || 0) * 100).toFixed(1)}% of samples with ${((metrics.precision || 0) * 100).toFixed(1)}% precision.</p>
      <p>Top contributing features: ${featureImportance.slice(0, 3).map(f => f.feature).join(', ')}.</p>
      ${sequenceStats ? `<p>Analyzed ${sequenceStats.total_sequences} ${sequenceStats.sequence_type} sequences with average length of ${sequenceStats.avg_length.toFixed(0)} base pairs.</p>` : ''}
    ` : `
      <p>The model achieved an R² score of ${(metrics.r2 || 0).toFixed(3)}, explaining ${((metrics.r2 || 0) * 100).toFixed(1)}% of variance.</p>
      <p>Top contributing features: ${featureImportance.slice(0, 3).map(f => f.feature).join(', ')}.</p>
    `}
  </div>
  
  <p style="margin-top: 40px; color: #71717a; font-size: 14px;">
    Generated by BioMLStudio • ${new Date().toLocaleString()}
  </p>
</body>
</html>`;
              
              const blob = new Blob([reportContent], { type: 'text/html' });
              const url = window.URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `report_job_${jobId}.html`;
              a.click();
            }}
            variant="secondary"
            size="lg"
          >
            Download Report
          </Button>
        </div>
      </div>
    </div>
    </>
  );
}
