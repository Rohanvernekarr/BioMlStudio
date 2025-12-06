'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Header } from '@/components/Header';
import { api } from '@/lib/api';
import { useAuth } from '@/lib/useAuth';
import { BarChart3, Rocket, Bot, FolderOpen, Folder, Dna, Search, TrendingUp, Target, FileText } from 'lucide-react';

interface DashboardStats {
  totalProjects: number;
  activeTraining: number;
  completedModels: number;
  totalDatasets: number;
  recentDatasets: any[];
  recentJobs: any[];
}

export default function Dashboard() {
  useAuth();
  const router = useRouter();
  
  const [stats, setStats] = useState<DashboardStats>({
    totalProjects: 0,
    activeTraining: 0,
    completedModels: 0,
    totalDatasets: 0,
    recentDatasets: [],
    recentJobs: []
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      // Load datasets
      const datasets = await api.request<any>('/api/v1/datasets/', {
        method: 'GET'
      });
      
      // Load jobs
      const jobs = await api.request<any>('/api/v1/jobs/', {
        method: 'GET'
      });

      setStats({
        totalProjects: jobs.items?.length || 0,
        activeTraining: jobs.items?.filter((j: any) => j.status === 'running' || j.status === 'queued').length || 0,
        completedModels: jobs.items?.filter((j: any) => j.status === 'completed').length || 0,
        totalDatasets: datasets.items?.length || 0,
        recentDatasets: datasets.items?.slice(0, 5) || [],
        recentJobs: jobs.items?.slice(0, 5) || []
      });
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-400';
      case 'running': return 'text-blue-400';
      case 'queued': return 'text-yellow-400';
      case 'failed': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <>
      <Header />
      <div className="min-h-[calc(100vh-64px)] bg-gradient-to-b from-black via-zinc-950/80 to-black">
        <div className="page-container padding-section">
          {/* Header */}
          <div className="section-spacing">
            <div className="text-center-wrapper mb-12">
              <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold mb-6 bg-gradient-to-br from-white via-zinc-100 to-zinc-400 bg-clip-text text-transparent">
                BioMLStudio
              </h1>
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-green-900/50 to-emerald-900/50 border border-green-500/30 rounded-full">
                <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                <span className="text-green-300 font-medium text-sm">No-Code AI Platform</span>
              </div>
              <p className="text-xl text-zinc-300 max-w-4xl mt-6 leading-relaxed">
                Full end-to-end, AI-powered, no-code Bioinformatics ML platform for researchers, students, and biologists
              </p>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 card-spacing">
              <Card className="p-6 bg-gradient-to-br from-zinc-900/80 to-zinc-800/80 border-zinc-700/50">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-zinc-400 text-sm font-medium mb-2">Active Projects</p>
                    <p className="text-3xl font-bold text-white">{stats.totalProjects}</p>
                  </div>
                  <BarChart3 className="w-8 h-8 text-zinc-400" />
                </div>
              </Card>

              <Card className="p-6 bg-gradient-to-br from-blue-900/40 to-blue-800/40 border-blue-700/50">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-blue-200 text-sm font-medium mb-2">Training Progress</p>
                    <p className="text-3xl font-bold text-blue-300">{stats.activeTraining}</p>
                  </div>
                  <Rocket className="w-8 h-8 text-blue-400" />
                </div>
              </Card>

              <Card className="p-6 bg-gradient-to-br from-green-900/40 to-green-800/40 border-green-700/50">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-green-200 text-sm font-medium mb-2">Trained Models</p>
                    <p className="text-3xl font-bold text-green-300">{stats.completedModels}</p>
                  </div>
                  <Bot className="w-8 h-8 text-green-400" />
                </div>
              </Card>

              <Card className="p-6 bg-gradient-to-br from-purple-900/40 to-purple-800/40 border-purple-700/50">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-purple-200 text-sm font-medium mb-2">Datasets</p>
                    <p className="text-3xl font-bold text-purple-300">{stats.totalDatasets}</p>
                  </div>
                  <FolderOpen className="w-8 h-8 text-purple-400" />
                </div>
              </Card>
            </div>

            {/* Core Modules - The 8 Essential Features */}
            <div className="section-spacing">
              <h2 className="text-3xl font-bold text-white mb-8 text-center">Core Platform Modules</h2>
              <div className="grid lg:grid-cols-2 xl:grid-cols-4 gap-6">
                {/* Module 1: Dataset Upload & Preprocessing */}
                <Card className="p-6 hover:bg-zinc-800/40 transition-all cursor-pointer group border-zinc-700/50 hover:border-zinc-600/70" 
                      onClick={() => router.push('/upload')}>
                  <div className="mb-4 group-hover:scale-110 transition-transform">
                    <Folder className="w-10 h-10 text-blue-400" />
                  </div>
                  <h3 className="text-white text-lg font-bold mb-3">Dataset Upload & Preprocessing</h3>
                  <p className="text-zinc-400 text-sm leading-relaxed">
                    FASTA/CSV upload with automatic preprocessing, quality reports, and biological validation
                  </p>
                </Card>

                {/* Module 2: Domain-Specific Pipelines */}
                <Card className="p-6 hover:bg-zinc-800/40 transition-all cursor-pointer group border-zinc-700/50 hover:border-zinc-600/70"
                      onClick={() => router.push('/pipelines')}>
                  <div className="mb-4 group-hover:scale-110 transition-transform">
                    <Dna className="w-10 h-10 text-green-400" />
                  </div>
                  <h3 className="text-white text-lg font-bold mb-3">Bioinformatics Pipelines</h3>
                  <p className="text-zinc-400 text-sm leading-relaxed">
                    Protein & DNA analysis with BioBERT/ESM2 embeddings and specialized feature extraction
                  </p>
                </Card>

                {/* Module 3: Automated ML Builder */}
                <Card className="p-6 hover:bg-zinc-800/40 transition-all cursor-pointer group border-zinc-700/50 hover:border-zinc-600/70"
                      onClick={() => router.push('/automl')}>
                  <div className="mb-4 group-hover:scale-110 transition-transform">
                    <Bot className="w-10 h-10 text-purple-400" />
                  </div>
                  <h3 className="text-white text-lg font-bold mb-3">AutoML Model Builder</h3>
                  <p className="text-zinc-400 text-sm leading-relaxed">
                    Auto model selection, hyperparameter tuning, and one-click training with live progress
                  </p>
                </Card>

                {/* Module 4: Model Explorer */}
                <Card className="p-6 hover:bg-zinc-800/40 transition-all cursor-pointer group border-zinc-700/50 hover:border-zinc-600/70"
                      onClick={() => router.push('/model-explorer')}>
                  <div className="mb-4 group-hover:scale-110 transition-transform">
                    <Search className="w-10 h-10 text-yellow-400" />
                  </div>
                  <h3 className="text-white text-lg font-bold mb-3">Model Explorer</h3>
                  <p className="text-zinc-400 text-sm leading-relaxed">
                    Visual model architecture, layer structure, parameter counts, and tensor shapes
                  </p>
                </Card>

                {/* Module 5: Results & Visualization Dashboard */}
                <Card className="p-6 hover:bg-zinc-800/40 transition-all cursor-pointer group border-zinc-700/50 hover:border-zinc-600/70"
                      onClick={() => router.push('/results')}>
                  <div className="mb-4 group-hover:scale-110 transition-transform">
                    <TrendingUp className="w-10 h-10 text-emerald-400" />
                  </div>
                  <h3 className="text-white text-lg font-bold mb-3">Results Dashboard</h3>
                  <p className="text-zinc-400 text-sm leading-relaxed">
                    Confusion matrices, ROC curves, feature importance, and attention visualizations
                  </p>
                </Card>

                {/* Module 6: Inference & Deployment */}
                <Card className="p-6 hover:bg-zinc-800/40 transition-all cursor-pointer group border-zinc-700/50 hover:border-zinc-600/70"
                      onClick={() => router.push('/inference')}>
                  <div className="mb-4 group-hover:scale-110 transition-transform">
                    <Target className="w-10 h-10 text-red-400" />
                  </div>
                  <h3 className="text-white text-lg font-bold mb-3">Inference Engine</h3>
                  <p className="text-zinc-400 text-sm leading-relaxed">
                    Upload new sequences, run predictions with confidence scores and biological interpretations
                  </p>
                </Card>

                {/* Module 7: Dataset Analysis Module */}
                <Card className="p-6 hover:bg-zinc-800/40 transition-all cursor-pointer group border-zinc-700/50 hover:border-zinc-600/70"
                      onClick={() => router.push('/datasets')}>
                  <div className="mb-4 group-hover:scale-110 transition-transform">
                    <BarChart3 className="w-10 h-10 text-cyan-400" />
                  </div>
                  <h3 className="text-white text-lg font-bold mb-3">Dataset Analysis</h3>
                  <p className="text-zinc-400 text-sm leading-relaxed">
                    Quality metrics, sequence statistics, and comprehensive data visualizations
                  </p>
                </Card>

                {/* Module 8: Test Cases & Reports */}
                <Card className="p-6 hover:bg-zinc-800/40 transition-all cursor-pointer group border-zinc-700/50 hover:border-zinc-600/70"
                      onClick={() => router.push('/reports')}>
                  <div className="mb-4 group-hover:scale-110 transition-transform">
                    <FileText className="w-10 h-10 text-orange-400" />
                  </div>
                  <h3 className="text-white text-lg font-bold mb-3">Reports & Testing</h3>
                  <p className="text-zinc-400 text-sm leading-relaxed">
                    Test cases, model comparison, training logs, and exportable reports
                  </p>
                </Card>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="grid lg:grid-cols-2 gap-8 section-spacing">
              {/* Recent Datasets */}
              <Card className="p-6 border-zinc-700/50">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-bold text-white">Recent Datasets</h3>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => router.push('/datasets')}
                  >
                    View All
                  </Button>
                </div>
                <div className="space-y-4">
                  {stats.recentDatasets.length > 0 ? (
                    stats.recentDatasets.map((dataset: any) => (
                      <div key={dataset.id} 
                           className="flex items-center justify-between p-4 bg-zinc-800/30 rounded-lg cursor-pointer hover:bg-zinc-800/50 transition-colors"
                           onClick={() => router.push(`/analysis/${dataset.id}`)}>
                        <div className="flex-1">
                          <p className="text-white font-medium">{dataset.name}</p>
                          <p className="text-zinc-400 text-sm capitalize">{dataset.dataset_type} • {new Date(dataset.created_at).toLocaleDateString()}</p>
                        </div>
                        <div className="text-xs text-zinc-500 bg-zinc-700/50 px-2 py-1 rounded">
                          {(dataset.file_size / 1024).toFixed(1)} KB
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="text-center py-8">
                      <Folder className="w-12 h-12 text-zinc-500 mx-auto mb-4" />
                      <p className="text-zinc-400 mb-4">No datasets uploaded yet</p>
                      <Button 
                        className="mt-4" 
                        size="sm"
                        onClick={() => router.push('/upload')}
                      >
                        Upload Dataset
                      </Button>
                    </div>
                  )}
                </div>
              </Card>

              {/* Training Progress */}
              <Card className="p-6 border-zinc-700/50">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-bold text-white">Training Progress</h3>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => router.push('/jobs')}
                  >
                    View All
                  </Button>
                </div>
                <div className="space-y-4">
                  {stats.recentJobs.length > 0 ? (
                    stats.recentJobs.map((job: any) => (
                      <div key={job.id} 
                           className="flex items-center justify-between p-4 bg-zinc-800/30 rounded-lg cursor-pointer hover:bg-zinc-800/50 transition-colors"
                           onClick={() => router.push(`/running/${job.id}`)}>
                        <div className="flex-1">
                          <p className="text-white font-medium">{job.name}</p>
                          <p className="text-zinc-400 text-sm">{job.job_type}</p>
                          {job.status === 'running' && job.progress && (
                            <div className="w-full bg-zinc-700 rounded-full h-2 mt-2">
                              <div 
                                className="bg-blue-400 h-2 rounded-full transition-all duration-300" 
                                style={{ width: `${job.progress}%` }}
                              />
                            </div>
                          )}
                        </div>
                        <div className={`text-xs px-2 py-1 rounded ${getStatusColor(job.status)} bg-zinc-700/50`}>
                          {job.status}
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="text-center py-8">
                      <Rocket className="w-12 h-12 text-zinc-500 mx-auto mb-4" />
                      <p className="text-zinc-400 mb-4">No training jobs yet</p>
                      <Button 
                        className="mt-4" 
                        size="sm"
                        onClick={() => router.push('/automl')}
                      >
                        Start Training
                      </Button>
                    </div>
                  )}
                </div>
              </Card>
            </div>
          </div>
        </div>
       </div>
    </>
  );
}