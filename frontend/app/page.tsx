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
      <div className="min-h-[calc(100vh-64px)] bg-gradient-to-b from-black via-zinc-950/80 to-black flex items-center justify-center">
        <div className="w-full max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center mb-12">
            <div className="mb-8">
              <h1 className="text-4xl sm:text-5xl lg:text-6xl xl:text-7xl font-bold mb-4 tracking-tight bg-gradient-to-br from-white via-zinc-100 to-zinc-400 bg-clip-text text-transparent">
                BioMLStudio
              </h1>

            </div>
            <p className="text-lg sm:text-xl text-zinc-300 leading-relaxed max-w-3xl mx-auto">
              Upload your dataset and get a trained ML model with comprehensive reports -
              <span className="text-white font-medium"> no coding required</span>.
            </p>
          </div>

          <Card className="max-w-3xl  mx-auto overflow-hidden">
            <div
              className={`relative border-2 border-dashed rounded-2xl p-8 sm:p-12 lg:p-16 text-center transition-all duration-300 group ${
                file 
                  ? 'border-emerald-400/60 bg-emerald-500/5 shadow-2xl' 
                  : 'border-zinc-600/40 hover:border-zinc-400/60 hover:bg-zinc-800/20 hover:shadow-xl cursor-pointer'
              }`}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
            >
              {/* Background Pattern */}
              <div className="absolute inset-0 opacity-5">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(255,255,255,0.1),transparent_70%)]" />
              </div>
              
              {!file ? (
                <div className="relative z-10">
                 
                  
                  {/* Main Text */}
                  <div className="mb-8">
                    <h3 className="text-2xl sm:text-3xl font-bold text-white mb-3">
                      Drop your file here
                    </h3>
                    <div className="flex items-center justify-center gap-4 mb-6">
                      <div className="h-px bg-zinc-600 flex-1 max-w-12" />
                      <span className="text-zinc-400 text-sm font-medium px-3">OR</span>
                      <div className="h-px bg-zinc-600 flex-1 max-w-12" />
                    </div>
                  </div>
                  
                  <div className="mb-8">
                    <label className="cursor-pointer inline-block group/btn">
                      <span className="inline-flex items-center gap-3  px-8 py-4 bg-gradient-to-r from-white to-zinc-100 text-black font-bold rounded-xl hover:from-zinc-50 hover:to-white active:scale-[0.98] transition-all duration-200 shadow-2xl hover:shadow-3xl text-base group-hover/btn:shadow-white/10">
                        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                          <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z" />
                        </svg>
                        Browse Files
                      </span>
                      <input
                        type="file"
                        accept=".csv,.fasta,.fa"
                        onChange={handleFileChange}
                        className="hidden"
                      />
                    </label>
                  </div>
                  
                  {/* File Types */}
                  <div className="flex flex-col sm:flex-row items-center justify-center gap-4 text-sm">
                    <div className="flex items-center gap-2 px-4 py-2 text-blue-300 rounded-lg border border-blue-500/20">
                      <span className="font-medium">CSV Files</span>
                    </div>
                    <div className="flex items-center gap-2 px-4 py-2  text-purple-300 rounded-lg border border-purple-500/20">
                      <span className="font-medium">FASTA Files</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="relative z-10">
                  {/* Success Icon */}
                  <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-emerald-500/20 to-emerald-600/20 rounded-2xl flex items-center justify-center border border-emerald-500/30">
                    <svg className="w-10 h-10 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  
                  {/* File Details */}
                  <div className="mb-6">
                    <h3 className="text-2xl font-bold text-white mb-2">File Ready</h3>
                    <p className="text-xl text-emerald-300 font-semibold mb-3">{file.name}</p>
                    <div className="flex flex-col sm:flex-row items-center justify-center gap-4 text-zinc-300">
                      <div className="flex items-center gap-2">
                        <span className="text-sm">📁</span>
                        <span className="text-sm font-medium">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                      </div>
                      {fileInfo && (
                        <div className="flex items-center gap-2">
                          <span className="text-sm">📊</span>
                          <span className="text-sm font-medium">{fileInfo.rows} rows, {fileInfo.cols} columns</span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {/* Remove Button */}
                  <button
                    onClick={() => setFile(null)}
                    className="inline-flex items-center gap-2 px-6 py-3 bg-red-500/10 hover:bg-red-500/20 text-red-400 hover:text-red-300 rounded-lg border border-red-500/30 hover:border-red-500/50 transition-all duration-200 font-medium"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1-1H8a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                    Remove File
                  </button>
                </div>
              )}
            </div>

            {error && (
              <div className="mt-8 p-6 bg-gradient-to-r from-red-950/50 to-red-900/50 border border-red-500/30 rounded-xl">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 bg-red-500/20 rounded-full flex items-center justify-center flex-shrink-0">
                    <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <p className="text-red-300 font-medium">{error}</p>
                </div>
              </div>
            )}

            {uploadedDatasetId && (
              <div className="mt-8 p-6 sm:p-8 bg-gradient-to-r from-green-950/40 to-emerald-950/40 border border-green-800/60 rounded-2xl shadow-xl">
                <p className="text-green-300 font-bold mb-4 text-xl flex items-center gap-2">
                  ✓ Upload successful!
                </p>
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
                    📊 Analyze Dataset
                  </Button>
                  <Button
                    onClick={() => router.push(`/configure/${uploadedDatasetId}`)}
                    size="lg"
                    className="flex-1"
                  >
                    🚀 Start Training
                  </Button>
                </div>
              </div>
            )}
            
            {!uploadedDatasetId && (
              <div className="mt-8 flex justify-center">
                <Button
                  onClick={handleUpload}
                  disabled={!file || uploading}
                  size="lg"
                  className="min-w-[280px] shadow-2xl hover:shadow-3xl disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {uploading ? (
                    <>
                      <svg className="w-5 h-5 mr-2 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Uploading...
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 12l2 2 4-4" />
                      </svg>
                      Upload Dataset
                    </>
                  )}
                </Button>
              </div>
            )}
          </Card>
        </div>
      </div>
    </>
  );
}
