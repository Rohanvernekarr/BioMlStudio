'use client';

import { useEffect, useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { Card } from '@/components/ui/Card';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { api } from '@/lib/api';
import { useAuth } from '@/lib/useAuth';
import { Job } from '@/types/api';

export default function RunningPage() {
  useAuth();
  const router = useRouter();
  const params = useParams();
  const jobId = Number(params.id);

  const [job, setJob] = useState<Job | null>(null);
  const [progress, setProgress] = useState(0);
  const [pollCount, setPollCount] = useState(0);
  const [steps, setSteps] = useState([
    { label: 'Data loaded', completed: false },
    { label: 'Preprocessing (cleaning, scaling)', completed: false },
    { label: 'Training models', completed: false },
    { label: 'Evaluating best model', completed: false },
  ]);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const jobData = await api.getJobStatus(jobId);
        setJob(jobData);
        setPollCount((prev) => prev + 1);

        if (jobData.status === 'completed') {
          setProgress(100);
          setSteps((prev) => prev.map((s) => ({ ...s, completed: true })));
          clearInterval(interval);
          setTimeout(() => router.push(`/results/${jobId}`), 1000);
        } else if (jobData.status === 'failed') {
          clearInterval(interval);
        } else if (jobData.status === 'running') {
          const currentProgress = jobData.progress || Math.min(progress + 5, 90);
          setProgress(currentProgress);

          if (currentProgress > 25) setSteps((prev) => prev.map((s, i) => i === 0 ? { ...s, completed: true } : s));
          if (currentProgress > 50) setSteps((prev) => prev.map((s, i) => i === 1 ? { ...s, completed: true } : s));
          if (currentProgress > 75) setSteps((prev) => prev.map((s, i) => i === 2 ? { ...s, completed: true } : s));
        } else {
          const fakeProgress = Math.min(20 + pollCount * 2, 90);
          setProgress(fakeProgress);
        }

        if (pollCount > 60) {
          clearInterval(interval);
        }
      } catch (err) {
        console.error('Failed to fetch job status', err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [jobId, progress, pollCount, router]);

  if (job?.status === 'failed') {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center p-8">
        <Card className="max-w-2xl w-full">
          <div className="text-center">
            <div className="text-4xl mb-4">✗</div>
            <h1 className="text-2xl font-bold mb-4">Analysis Failed</h1>
            <p className="text-zinc-400 mb-6">
              {job.error_message || 'An error occurred during analysis'}
            </p>
            <button
              onClick={() => router.push('/')}
              className="px-6 py-2 bg-white text-black rounded-lg hover:bg-zinc-200"
            >
              Start New Analysis
            </button>
          </div>
        </Card>
      </div>
    );
  }

  if (pollCount > 60) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center p-8">
        <Card className="max-w-2xl w-full">
          <div className="text-center">
            <div className="text-4xl mb-4">⏱️</div>
            <h1 className="text-2xl font-bold mb-4">Taking Longer Than Expected</h1>
            <p className="text-zinc-400 mb-4">
              The analysis is still processing. Current status: <strong>{job?.status || 'unknown'}</strong>
            </p>
            <p className="text-sm text-zinc-500 mb-6">
              Note: Celery workers might not be running. Check backend logs.
            </p>
            <div className="flex gap-4 justify-center">
              <button
                onClick={() => router.push('/')}
                className="px-6 py-2 bg-zinc-800 text-white rounded-lg hover:bg-zinc-700 border border-zinc-700"
              >
                Start New Analysis
              </button>
              <button
                onClick={() => window.location.reload()}
                className="px-6 py-2 bg-white text-black rounded-lg hover:bg-zinc-200"
              >
                Refresh
              </button>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-8">
      <div className="max-w-2xl w-full">
        <Card>
          <div className="text-center mb-8">
            <div className="text-4xl mb-6">🔬</div>
            <h1 className="text-3xl font-bold mb-2">Running Analysis</h1>
            <p className="text-zinc-400 mb-2">
              Status: <span className="text-white capitalize">{job?.status || 'loading'}</span>
            </p>
            <p className="text-sm text-zinc-500">
              {job?.status === 'queued' && 'Waiting in queue...'}
              {job?.status === 'pending' && 'Initializing...'}
              {job?.status === 'running' && 'Processing your data...'}
            </p>
          </div>

          <ProgressBar value={progress} className="mb-8" />

          <div className="space-y-4">
            {steps.map((step, i) => (
              <div key={i} className="flex items-center gap-3">
                <div
                  className={`w-6 h-6 rounded-full flex items-center justify-center ${
                    step.completed
                      ? 'bg-white text-black'
                      : 'bg-zinc-800 text-zinc-400'
                  }`}
                >
                  {step.completed ? '✓' : i === steps.findIndex((s) => !s.completed) ? '⏳' : '○'}
                </div>
                <span className={step.completed ? 'text-white' : 'text-zinc-400'}>
                  {step.label}
                </span>
              </div>
            ))}
          </div>

          <div className="mt-8 text-center text-sm text-zinc-500">
            Job ID: {jobId}
          </div>
        </Card>
      </div>
    </div>
  );
}
