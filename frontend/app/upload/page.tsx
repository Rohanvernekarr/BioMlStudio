'use client';

import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Header } from '@/components/Header';
import { api } from '@/lib/api';
import { useAuth } from '@/lib/useAuth';
import { Folder, Dna, BarChart3, FileSpreadsheet, Image, Search, Calculator, Ruler, Target, TrendingUp, CheckCircle2, FileText, Rocket, Bot } from 'lucide-react';

export default function Upload() {
  useAuth();
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [fileInfo, setFileInfo] = useState<any>(null);
  const [uploadedDatasetId, setUploadedDatasetId] = useState<number | null>(null);
  const [preprocessing, setPreprocessing] = useState(false);

  // Check for existing uploaded dataset on page load
  useEffect(() => {
    const lastDataset = localStorage.getItem('lastUploadedDataset');
    if (lastDataset) {
      try {
        const datasetInfo = JSON.parse(lastDataset);
        // Check if uploaded within last 24 hours
        const uploadTime = new Date(datasetInfo.uploadedAt);
        const now = new Date();
        const hoursDiff = (now.getTime() - uploadTime.getTime()) / (1000 * 60 * 60);
        
        if (hoursDiff < 24) {
          setUploadedDatasetId(datasetInfo.id);
          setFileInfo({
            name: datasetInfo.name,
            type: datasetInfo.fileType,
            sizeFormatted: formatFileSize(datasetInfo.size)
          });
        } else {
          // Clear old dataset info
          localStorage.removeItem('lastUploadedDataset');
        }
      } catch (error) {
        console.error('Error loading saved dataset:', error);
      }
    }
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      validateAndSetFile(selectedFile);
    }
  };

  const validateAndSetFile = (selectedFile: File) => {
    const fileName = selectedFile.name.toLowerCase();
    const validExtensions = ['.csv', '.fasta', '.fa', '.fas', '.xlsx', '.png', '.jpg', '.jpeg'];
    const isValid = validExtensions.some(ext => fileName.endsWith(ext));
    
    if (!isValid) {
      setError('Please upload a supported file: CSV, FASTA, Excel, or Image files');
      return;
    }
    
    setFile(selectedFile);
    setError('');
    setUploadedDatasetId(null);
    
    // Auto-detect file type and show info
    const fileType = detectFileType(fileName);
    setFileInfo({
      name: selectedFile.name,
      size: selectedFile.size,
      type: fileType,
      sizeFormatted: formatFileSize(selectedFile.size)
    });
  };

  const detectFileType = (fileName: string) => {
    if (fileName.endsWith('.csv')) return 'CSV Data';
    if (fileName.endsWith('.fasta') || fileName.endsWith('.fa') || fileName.endsWith('.fas')) return 'FASTA Sequence';
    if (fileName.endsWith('.xlsx')) return 'Excel Data';
    if (fileName.endsWith('.png') || fileName.endsWith('.jpg') || fileName.endsWith('.jpeg')) return 'Image Data';
    return 'Unknown';
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      validateAndSetFile(droppedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setPreprocessing(false);
    setError('');

    try {
      // Auto-detect dataset type
      const fileName = file.name.toLowerCase();
      let datasetType = 'general';
      
      if (fileName.endsWith('.fasta') || fileName.endsWith('.fa') || fileName.endsWith('.fas')) {
        datasetType = 'dna'; // Will be enhanced to detect protein vs DNA
      }
      
      setPreprocessing(true);
      const dataset = await api.uploadDataset(file, file.name, datasetType);
      
      // Store dataset info in localStorage for cross-page availability
      const datasetInfo = {
        id: dataset.id,
        name: dataset.name || file.name,
        type: datasetType,
        uploadedAt: new Date().toISOString(),
        size: file.size,
        fileType: fileInfo?.type || detectFileType(file.name)
      };
      localStorage.setItem('lastUploadedDataset', JSON.stringify(datasetInfo));
      localStorage.setItem('availableDatasets', JSON.stringify([datasetInfo]));
      
      setUploadedDatasetId(dataset.id);
      setUploading(false);
      setPreprocessing(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setUploading(false);
      setPreprocessing(false);
    }
  };

  const resetUpload = () => {
    setFile(null);
    setUploadedDatasetId(null);
    setFileInfo(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <>
      <Header />
      <div className="min-h-[calc(100vh-64px)] bg-gradient-to-b from-black via-zinc-950/80 to-black">
        <div className="page-container padding-section">
          {/* Header */}
          <div className="section-spacing text-center-wrapper">
            <Button
              variant="ghost"
              onClick={() => router.back()}
              className="mb-6 self-start"
            >
              ← Back to Dashboard
            </Button>
            <h1 className="text-5xl sm:text-6xl font-bold mb-6 bg-gradient-to-br from-white via-zinc-100 to-zinc-400 bg-clip-text text-transparent">
              Dataset Upload & Preprocessing
            </h1>
            <p className="text-xl text-zinc-300 max-w-4xl leading-relaxed">
              Upload your biological datasets with automatic preprocessing, quality analysis, and format validation
            </p>
          </div>

          {!uploadedDatasetId ? (
            <>
              {/* Upload Card */}
              <Card className="centered-card-xl card-spacing overflow-hidden">
                <div
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  onClick={() => fileInputRef.current?.click()}
                  className={`relative border-2 border-dashed rounded-2xl p-10 sm:p-14 lg:p-20 text-center cursor-pointer transition-all duration-300 ${
                    file 
                      ? 'border-emerald-400/60 bg-emerald-500/5 shadow-2xl' 
                      : 'border-zinc-600/40 hover:border-zinc-400/60 hover:bg-zinc-800/20 hover:shadow-xl'
                  }`}
                >
                  {!file ? (
                    <div className="relative z-10">
                      <Folder className="w-16 h-16 text-zinc-400 mx-auto mb-6" />
                      <h3 className="text-3xl font-bold text-white mb-4">
                        Drop your dataset here
                      </h3>
                      <div className="flex items-center justify-center gap-4 mb-6">
                        <div className="h-px bg-zinc-600 flex-1 max-w-12" />
                        <span className="text-zinc-400 text-sm font-medium px-3">OR</span>
                        <div className="h-px bg-zinc-600 flex-1 max-w-12" />
                      </div>
                      <p className="text-xl text-zinc-200 mb-8">Click to browse files</p>
                      <p className="text-zinc-400 text-sm">
                        Supports CSV, FASTA, Excel, and Image files (Max 100MB)
                      </p>
                      <input
                      title='DS'
                        ref={fileInputRef}
                        type="file"
                        onChange={handleFileChange}
                        accept=".csv,.fasta,.fa,.fas,.xlsx,.png,.jpg,.jpeg"
                        className="hidden"
                      />
                    </div>
                  ) : (
                    <div className="relative z-10">
                      <CheckCircle2 className="w-12 h-12 text-emerald-400 mx-auto mb-6" />
                      <h3 className="text-2xl font-bold text-white mb-2">File Ready</h3>
                      <p className="text-xl text-emerald-300 font-semibold mb-4">{fileInfo.name}</p>
                      <div className="flex flex-col sm:flex-row items-center justify-center gap-4 text-zinc-300 mb-6">
                        <div className="flex items-center gap-2 px-3 py-1 bg-zinc-700/50 rounded-lg">
                          <span className="text-sm font-medium">{fileInfo.type}</span>
                        </div>
                        <div className="flex items-center gap-2 px-3 py-1 bg-zinc-700/50 rounded-lg">
                          <span className="text-sm font-medium">{fileInfo.sizeFormatted}</span>
                        </div>
                      </div>
                      <Button
                        onClick={resetUpload}
                        variant="outline"
                        size="sm"
                      >
                        Remove File
                      </Button>
                    </div>
                  )}
                </div>

                {/* Error Message */}
                {error && (
                  <div className="component-spacing p-6 bg-gradient-to-r from-red-950/50 to-red-900/50 border border-red-500/30 rounded-xl">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-red-500/20 rounded-full flex items-center justify-center shrink-0">
                        <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </div>
                      <p className="text-red-300 font-medium">{error}</p>
                    </div>
                  </div>
                )}

                {/* Upload Button */}
                {file && (
                  <div className="component-spacing flex justify-center gap-4">
                    <Button
                      onClick={handleUpload}
                      disabled={uploading}
                      size="lg"
                      className="min-w-[280px]"
                    >
                      {uploading ? (
                        preprocessing ? (
                          <>
                            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mr-3"></div>
                            Processing & Analyzing...
                          </>
                        ) : (
                          <>
                            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white mr-3"></div>
                            Uploading...
                          </>
                        )
                      ) : (
                        'Upload & Process Dataset'
                      )}
                    </Button>
                  </div>
                )}
              </Card>

              {/* Supported Formats */}
              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 card-spacing">
                <Card className="p-6 text-center bg-blue-900/20 border-blue-700/30">
                  <BarChart3 className="w-10 h-10 text-blue-400 mx-auto mb-3" />
                  <h3 className="text-white font-bold mb-2">CSV Data</h3>
                  <p className="text-blue-200 text-sm">Tabular data with automatic column detection</p>
                </Card>

                <Card className="p-6 text-center bg-green-900/20 border-green-700/30">
                  <Dna className="w-10 h-10 text-green-400 mx-auto mb-3" />
                  <h3 className="text-white font-bold mb-2">FASTA Sequences</h3>
                  <p className="text-green-200 text-sm">DNA, RNA, and protein sequences</p>
                </Card>

                <Card className="p-6 text-center bg-purple-900/20 border-purple-700/30">
                  <FileSpreadsheet className="w-10 h-10 text-purple-400 mx-auto mb-3" />
                  <h3 className="text-white font-bold mb-2">Excel Files</h3>
                  <p className="text-purple-200 text-sm">Spreadsheet data with multiple sheets</p>
                </Card>

                <Card className="p-6 text-center bg-orange-900/20 border-orange-700/30">
                  <Image className="w-10 h-10 text-orange-400 mx-auto mb-3" />
                  <h3 className="text-white font-bold mb-2">Images</h3>
                  <p className="text-orange-200 text-sm">Microscopy and biological imagery</p>
                </Card>
              </div>

              {/* Preprocessing Features */}
              <Card className="p-8 border-zinc-700/50 card-spacing">
                <h2 className="text-2xl font-bold text-white mb-8 text-center">Automatic Preprocessing Features</h2>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mt-4">
                  <div className="text-center">
                    <Search className="w-10 h-10 text-blue-400 mx-auto mb-3" />
                    <h3 className="text-white font-bold mb-2">Quality Analysis</h3>
                    <p className="text-zinc-400 text-sm">Sequence quality metrics, missing data detection, and completeness analysis</p>
                  </div>

                  <div className="text-center">
                    <Calculator className="w-10 h-10 text-green-400 mx-auto mb-3" />
                    <h3 className="text-white font-bold mb-2">Feature Extraction</h3>
                    <p className="text-zinc-400 text-sm">K-mer representation, one-hot encoding, and BioBERT embeddings</p>
                  </div>

                  <div className="text-center">
                    <Ruler className="w-10 h-10 text-purple-400 mx-auto mb-3" />
                    <h3 className="text-white font-bold mb-2">Normalization</h3>
                    <p className="text-zinc-400 text-sm">Sequence padding, truncation, and length standardization</p>
                  </div>

                  <div className="text-center">
                    <BarChart3 className="w-10 h-10 text-yellow-400 mx-auto mb-3" />
                    <h3 className="text-white font-bold mb-2">Statistics</h3>
                    <p className="text-zinc-400 text-sm">Distribution analysis, class balance, and summary metrics</p>
                  </div>

                  <div className="text-center">
                    <Target className="w-10 h-10 text-red-400 mx-auto mb-3" />
                    <h3 className="text-white font-bold mb-2">Validation</h3>
                    <p className="text-zinc-400 text-sm">Format validation, biological sequence verification</p>
                  </div>

                  <div className="text-center">
                    <TrendingUp className="w-10 h-10 text-cyan-400 mx-auto mb-3" />
                    <h3 className="text-white font-bold mb-2">Visualization</h3>
                    <p className="text-zinc-400 text-sm">Length distributions, composition plots, and quality reports</p>
                  </div>
                </div>
              </Card>
            </>
          ) : (
            // Success State
            <Card className="centered-card-xl">
              <div className="p-12 text-center">
                <CheckCircle2 className="w-16 h-16 text-emerald-400 mx-auto mb-6" />
                <h2 className="text-3xl font-bold text-white mb-4">Dataset Ready Across All Modules!</h2>
                <p className="text-xl text-zinc-300 mb-4 max-w-2xl mx-auto leading-relaxed">
                  Your dataset has been uploaded and is now available in Pipelines, AutoML, and all analysis modules.
                </p>
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-green-900/50 border border-green-500/30 rounded-full mb-6">
                  <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                  <span className="text-green-300 font-medium text-sm">Dataset synchronized across platform</span>
                </div>
                
                {/* Next Steps - Biology-First Approach */}
                <div className="mb-8">
                  <h3 className="text-xl font-bold text-white mb-4 text-center">
                    🧬 Choose Your Analysis Path
                    {fileInfo?.type === 'FASTA Sequence' && 
                      <span className="block text-sm text-green-400 font-normal mt-1">
                        ⭐ Recommended for sequence data
                      </span>
                    }
                  </h3>
                  
                  {/* Biology-Specific Pipelines (Primary Options) */}
                  <div className="grid md:grid-cols-2 gap-4 mb-6">
                    <Card className="p-6 border-blue-500/30 bg-blue-900/10 hover:bg-blue-900/20 transition-all cursor-pointer group">
                      <div className="text-center w-full h-full" onClick={() => router.push('/pipelines')}>
                        <Dna className="w-12 h-12 text-blue-400 mx-auto mb-3 group-hover:scale-110 transition-transform" />
                        <h4 className="text-lg font-bold text-white mb-2">Domain-Specific Pipelines</h4>
                        <p className="text-blue-200 text-sm mb-4">
                          {fileInfo?.type === 'FASTA Sequence' 
                            ? 'Perfect for your sequence data! Start with protein or DNA analysis.'
                            : 'Biology-focused AI workflows for specialized analysis.'}
                        </p>
                        <Button size="sm" className="w-full">
                          Choose Pipeline →
                        </Button>
                      </div>
                    </Card>

                    <Card className="p-6 border-purple-500/30 bg-purple-900/10 hover:bg-purple-900/20 transition-all cursor-pointer group">
                      <div className="text-center w-full h-full" onClick={() => router.push('/automl')}>
                        <Bot className="w-12 h-12 text-purple-400 mx-auto mb-3 group-hover:scale-110 transition-transform" />
                        <h4 className="text-lg font-bold text-white mb-2">AutoML Builder</h4>
                        <p className="text-purple-200 text-sm mb-4">
                          Build custom AI models with automated optimization and training.
                        </p>
                        <Button variant="outline" size="sm" className="w-full">
                          Build Custom Model
                        </Button>
                      </div>
                    </Card>
                  </div>

                  {/* Direct Action Buttons */}
                  <div className="flex flex-col sm:flex-row gap-3 justify-center mb-4">
                    <Button
                      onClick={() => router.push(`/datasets/${uploadedDatasetId}`)}
                      variant="outline"
                      size="lg"
                      className="flex items-center gap-2"
                    >
                      <BarChart3 className="w-4 h-4" />
                      View Analysis Report
                    </Button>
                    <Button
                      onClick={() => router.push(`/automl?dataset=${uploadedDatasetId}`)}
                      size="lg"
                      className="flex items-center gap-2"
                    >
                      <Rocket className="w-4 h-4" />
                      Start Training Model
                    </Button>
                  </div>
                  
                  {/* Secondary Navigation */}
                  <div className="flex flex-col sm:flex-row gap-2 justify-center">
                    <Button
                      onClick={() => router.push('/dashboard')}
                      variant="ghost"
                      size="sm"
                      className="flex items-center gap-2 text-zinc-400"
                    >
                      <FileText className="w-4 h-4" />
                      Back to Dashboard
                    </Button>
                  </div>
                </div>

                <Button
                  onClick={resetUpload}
                  variant="ghost"
                  className="text-zinc-400 hover:text-white"
                >
                  Upload Another Dataset
                </Button>
              </div>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}