'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Header } from '@/components/Header';
import { api } from '@/lib/api';
import { useAuth } from '@/lib/useAuth';

export default function Home() {
  useAuth();
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [fileInfo, setFileInfo] = useState<{ rows: number; cols: number } | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (!selectedFile.name.endsWith('.csv')) {
        setError('Please upload a CSV file');
        return;
      }
      setFile(selectedFile);
      setError('');
      setFileInfo(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.name.endsWith('.csv')) {
      setFile(droppedFile);
      setError('');
      setFileInfo(null);
    } else {
      setError('Please upload a CSV file');
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError('');

    try {
      const dataset = await api.uploadDataset(file, file.name);
      router.push(`/configure/${dataset.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setUploading(false);
    }
  };

  return (
    <>
      <Header />
      <div className="min-h-screen bg-black flex items-center justify-center p-8">
        <div className="max-w-2xl w-full">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">BioMLStudio</h1>
          <p className="text-xl text-zinc-400">
            Upload your CSV and get a trained model and report, no coding.
          </p>
        </div>

        <Card>
          <div
            className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
              file ? 'border-white bg-zinc-800' : 'border-zinc-700 hover:border-zinc-600'
            }`}
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
          >
            {!file ? (
              <>
                <svg
                  className="mx-auto h-12 w-12 text-zinc-400 mb-4"
                  stroke="currentColor"
                  fill="none"
                  viewBox="0 0 48 48"
                >
                  <path
                    d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <p className="text-lg mb-2">Drop your CSV file here</p>
                <p className="text-sm text-zinc-400 mb-4">or</p>
                <label className="cursor-pointer">
                  <span className="px-4 py-2 bg-white text-black rounded-lg hover:bg-zinc-200 transition-colors inline-block">
                    Browse File
                  </span>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>
                <p className="text-sm text-zinc-500 mt-4">
                  Accepts CSV with samples as rows, features as columns
                </p>
              </>
            ) : (
              <>
                <div className="text-2xl mb-2">✓</div>
                <p className="text-lg font-medium mb-1">{file.name}</p>
                <p className="text-sm text-zinc-400">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
                {fileInfo && (
                  <p className="text-sm text-zinc-400 mt-2">
                    {fileInfo.rows} rows, {fileInfo.cols} columns
                  </p>
                )}
                <button
                  onClick={() => setFile(null)}
                  className="text-sm text-zinc-400 hover:text-white mt-4"
                >
                  Remove
                </button>
              </>
            )}
          </div>

          {error && (
            <div className="mt-4 p-3 bg-red-900/20 border border-red-800 rounded-lg text-red-400">
              {error}
            </div>
          )}

          <div className="mt-6 flex justify-end">
            <Button
              onClick={handleUpload}
              disabled={!file || uploading}
              size="lg"
            >
              {uploading ? 'Uploading...' : 'Next: Configure Analysis'}
            </Button>
          </div>
        </Card>
      </div>
    </div>
    </>
  );
}
