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
  const [uploadedDatasetId, setUploadedDatasetId] = useState<number | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      const fileName = selectedFile.name.toLowerCase();
      if (!fileName.endsWith('.csv') && !fileName.endsWith('.fasta') && !fileName.endsWith('.fa')) {
        setError('Please upload a CSV or FASTA file');
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
    if (droppedFile) {
      const fileName = droppedFile.name.toLowerCase();
      if (fileName.endsWith('.csv') || fileName.endsWith('.fasta') || fileName.endsWith('.fa')) {
        setFile(droppedFile);
        setError('');
        setFileInfo(null);
      } else {
        setError('Please upload a CSV or FASTA file');
      }
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError('');

    try {
      // Auto-detect FASTA files and set dataset_type
      const fileName = file.name.toLowerCase();
      const isFasta = fileName.endsWith('.fasta') || fileName.endsWith('.fa');
      const datasetType = isFasta ? 'dna' : 'general';
      
      const dataset = await api.uploadDataset(file, file.name, datasetType);
      setUploadedDatasetId(dataset.id);
      setUploading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setUploading(false);
    }
  };

  return (
    <>
      <Header />
      <div className="min-h-[calc(100vh-73px)] bg-gradient-to-b from-black via-zinc-950 to-black flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl w-full">
        <div className="text-center mb-16">
          <h1 className="text-6xl sm:text-7xl font-bold mb-8 tracking-tight bg-gradient-to-br from-white via-zinc-100 to-zinc-500 bg-clip-text text-transparent">
            BioMLStudio
          </h1>
          <p className="text-xl sm:text-2xl text-zinc-400 leading-relaxed max-w-3xl mx-auto font-light">
            Upload your dataset and get a trained ML model with comprehensive reports—no coding required.
          </p>
        </div>

        <Card className="!p-10 sm:!p-12">
          <div
            className={`border-2 border-dashed rounded-2xl p-20 text-center transition-all duration-300 ${
              file ? 'border-white bg-white/5 shadow-2xl' : 'border-zinc-700 hover:border-zinc-500 hover:bg-zinc-900/50 hover:shadow-xl'
            }`}
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
          >
            {!file ? (
              <>
                <svg
                  className="mx-auto h-16 w-16 text-zinc-400 mb-6"
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
                <p className="text-2xl mb-4 font-semibold">Drop your file here</p>
                <p className="text-sm text-zinc-500 mb-8">or</p>
                <label className="cursor-pointer">
                  <span className="px-8 py-4 bg-white text-black font-semibold rounded-xl hover:bg-zinc-100 active:scale-95 transition-all duration-200 inline-block shadow-2xl text-lg">
                    📂 Browse Files
                  </span>
                  <input
                    type="file"
                    accept=".csv,.fasta,.fa"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </label>
                <p className="text-sm text-zinc-400 mt-8 font-medium">
                  📊 CSV for tabular data • 🧬 FASTA for DNA/protein sequences
                </p>
              </>
            ) : (
              <>
                <div className="text-5xl mb-4">✓</div>
                <p className="text-xl font-semibold mb-2">{file.name}</p>
                <p className="text-base text-zinc-400 font-medium">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
                {fileInfo && (
                  <p className="text-base text-zinc-300 mt-3 font-medium">
                    {fileInfo.rows} rows, {fileInfo.cols} columns
                  </p>
                )}
                <button
                  onClick={() => setFile(null)}
                  className="text-base text-zinc-400 hover:text-white mt-6 font-medium transition-colors"
                >
                  ❌ Remove
                </button>
              </>
            )}
          </div>

          {error && (
            <div className="mt-8 p-5 bg-red-950/40 border-2 border-red-800/60 rounded-2xl text-red-300 text-base font-medium shadow-xl">
              <span className="inline-block mr-2">⚠️</span>
              {error}
            </div>
          )}

          {uploadedDatasetId && (
            <div className="mt-8 p-8 bg-green-950/40 border-2 border-green-800/60 rounded-2xl shadow-xl">
              <p className="text-green-300 font-bold mb-4 text-xl">✓ Upload successful!</p>
              <p className="text-base text-zinc-300 mb-6 leading-relaxed">
                Would you like to analyze your dataset first or skip directly to model training?
              </p>
              <div className="flex flex-col sm:flex-row gap-4">
                <Button
                  onClick={() => router.push(`/analysis/${uploadedDatasetId}`)}
                  variant="outline"
                  size="lg"
                  className="flex-1"
                >
                  Analyze Dataset
                </Button>
                <Button
                  onClick={() => router.push(`/configure/${uploadedDatasetId}`)}
                  size="lg"
                  className="flex-1"
                >
                  Start Training
                </Button>
              </div>
            </div>
          )}

          {!uploadedDatasetId && (
            <div className="mt-10 flex justify-end">
              <Button
                onClick={handleUpload}
                disabled={!file || uploading}
                size="lg"
                className="min-w-[240px] !py-4 !text-lg font-semibold"
              >
                {uploading ? '⏳ Uploading...' : '🚀 Upload Dataset'}
              </Button>
            </div>
          )}
        </Card>
      </div>
    </div>
    </>
  );
}
