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
      const [jobData, resultsData] = await Promise.all([
        api.getJobStatus(jobId),
        api.getJobResults(jobId),
      ]);

      setJob(jobData);
      setResults(resultsData.results);
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

  const isClassification = metrics.accuracy !== undefined;
  const maxImportance = featureImportance.length > 0 ? Math.max(...featureImportance.map((f) => f.importance)) : 1;

  return (
    <>
      <Header />
      <div className="min-h-screen bg-black p-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-8">
            <h1 className="text-3xl font-bold mb-2">Results</h1>
            <p className="text-zinc-400">"
            {job?.name} • Best model: {results.best_model || 'RandomForest'}
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          {isClassification ? (
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

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <Card>
            <h2 className="text-xl font-bold mb-4">Top Features</h2>
            <p className="text-sm text-zinc-400 mb-4">
              Features influencing prediction the most
            </p>
            <div className="space-y-3">
              {featureImportance.slice(0, 10).map((item, i) => (
                <div key={i}>
                  <div className="flex justify-between text-sm mb-1">
                    <span>{item.feature}</span>
                    <span className="text-zinc-400">{item.importance.toFixed(3)}</span>
                  </div>
                  <div className="w-full bg-zinc-800 rounded-full h-2">
                    <div
                      className="bg-white h-2 rounded-full"
                      style={{ width: `${(item.importance / maxImportance) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {isClassification && confusionMatrix.length > 0 && (
            <Card>
              <h2 className="text-xl font-bold mb-4">Confusion Matrix</h2>
              <p className="text-sm text-zinc-400 mb-4">
                Actual vs predicted classifications
              </p>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <tbody>
                    {confusionMatrix.map((row, i) => (
                      <tr key={i}>
                        {row.map((cell, j) => {
                          const max = Math.max(...confusionMatrix.flat());
                          const intensity = cell / max;
                          return (
                            <td
                              key={j}
                              className="p-4 text-center border border-zinc-800"
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

        <Card>
          <h2 className="text-xl font-bold mb-4">Summary</h2>
          <ul className="space-y-2 text-zinc-300 list-disc list-inside">
            {isClassification ? (
              <>
                <li>
                  The model correctly classified {((metrics.accuracy || 0) * 100).toFixed(1)}% of
                  samples.
                </li>
                <li>
                  Top contributing features: {featureImportance.slice(0, 3).map((f) => f.feature).join(', ')}.
                </li>
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

        <div className="mt-8 flex gap-4">
          <Button onClick={() => router.push('/')} size="lg">
            Run New Analysis
          </Button>
        </div>
      </div>
    </div>
    </>
  );
}
