'use client';

import { useEffect, useState } from 'react';
import { useRouter, useParams } from 'next/navigation';
import { Button } from '@/components/ui/Button';
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
        const jobData = await api.getWorkflowStatus(jobId);
        setJob(jobData as any);
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
      <div className="min-h-screen bg-black flex items-center justify-center px-4 sm:px-6 lg:px-8 py-16">
        <Card className="max-w-2xl w-full">
          <div className="text-center">
            <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-red-950/30 border-2 border-red-800/50 flex items-center justify-center">
              <span className="text-4xl">✗</span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-bold mb-4">Analysis Failed</h1>
            <p className="text-zinc-400 mb-8 leading-relaxed">
              {job.error_message || 'An error occurred during analysis'}
            </p>
            <Button
              onClick={() => router.push('/')}
              size="lg"
            >
              Start New Analysis
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  if (pollCount > 60) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center px-4 sm:px-6 lg:px-8 py-16">
        <Card className="max-w-2xl w-full">
          <div className="text-center">
            <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-zinc-900 border-2 border-zinc-800 flex items-center justify-center">
              <span className="text-4xl">⏱️</span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-bold mb-4">Taking Longer Than Expected</h1>
            <p className="text-zinc-400 mb-3 leading-relaxed">
              The analysis is still processing. Current status: <strong className="text-white">{job?.status || 'unknown'}</strong>
            </p>
            <p className="text-sm text-zinc-500 mb-8">
              Note: Celery workers might not be running. Check backend logs.
            </p>
            <div className="flex gap-4 justify-center">
              <Button
                variant="secondary"
                onClick={() => router.push('/')}
                size="lg"
              >
                Start New Analysis
              </Button>
              <Button
                onClick={() => window.location.reload()}
                size="lg"
              >
                Refresh
              </Button>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black flex items-center justify-center px-4 sm:px-6 lg:px-8 py-16">
      <div className="max-w-2xl w-full">
        <Card>
          <div className="text-center mb-10">
            <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-zinc-900 border-2 border-zinc-800 flex items-center justify-center animate-pulse">
              <span className="text-4xl">🔬</span>
            </div>
            <h1 className="text-4xl sm:text-5xl font-bold mb-4 tracking-tight">Running Analysis</h1>
            <p className="text-zinc-400 mb-2 text-lg">
              Status: <span className="text-white capitalize font-medium">{job?.status || 'loading'}</span>
            </p>
            <p className="text-sm text-zinc-500">
              {job?.status === 'queued' && 'Waiting in queue...'}
              {job?.status === 'pending' && 'Initializing...'}
              {job?.status === 'running' && 'Processing your data...'}
            </p>
          </div>

          <ProgressBar value={progress} className="mb-10" />

          <div className="space-y-3">{steps.map((step, i) => (
              <div key={i} className="flex items-center gap-4 p-4 rounded-xl bg-zinc-950/50 border border-zinc-800/50">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-all ${
                    step.completed
                      ? 'bg-white text-black'
                      : 'bg-zinc-800 text-zinc-400'
                  }`}
                >
                  {step.completed ? '✓' : i === steps.findIndex((s) => !s.completed) ? '⏳' : '○'}
                </div>
                <span className={`${step.completed ? 'text-white font-medium' : 'text-zinc-400'}`}>
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
