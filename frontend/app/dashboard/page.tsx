'use client';

import ProtectedRoute from '@/components/ProtectedRoute';
import { useAuth } from '@/context/AuthContext';

function DashboardContent() {
  const { token, logout } = useAuth();

  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <h1 className="text-xl font-bold text-gray-900">BioMLStudio</h1>
              </div>
            </div>
            <div className="hidden sm:ml-6 sm:flex sm:items-center">
              <button
                onClick={logout}
                className="px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                Sign out
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="py-10">
        <header>
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          </div>
        </header>
        <main>
          <div className="max-w-7xl mx-auto sm:px-6 lg:px-8">
            <div className="px-4 py-8 sm:px-0">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Upload Dataset */}
                <a href="/datasets/upload" className="group">
                  <div className="bg-white rounded-xl shadow-sm hover:shadow-md transition-all duration-200 p-6 border border-gray-100 hover:border-blue-200">
                    <div className="flex items-center mb-3">
                      <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center group-hover:bg-blue-200 transition-colors">
                        <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-gray-800 ml-3">Upload Dataset</h3>
                    </div>
                    <p className="text-gray-600 text-sm">Upload FASTA/FASTQ or CSV/TSV files for analysis and training.</p>
                  </div>
                </a>

                {/* Datasets */}
                <a href="/datasets" className="group">
                  <div className="bg-white rounded-xl shadow-sm hover:shadow-md transition-all duration-200 p-6 border border-gray-100 hover:border-indigo-200">
                    <div className="flex items-center mb-3">
                      <div className="w-10 h-10 bg-indigo-100 rounded-lg flex items-center justify-center group-hover:bg-indigo-200 transition-colors">
                        <svg className="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7h18M3 12h18M3 17h18" />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-gray-800 ml-3">My Datasets</h3>
                    </div>
                    <p className="text-gray-600 text-sm">Browse, preview, and manage your uploaded datasets.</p>
                  </div>
                </a>

                {/* Create Training Job */}
                <a href="/jobs/create" className="group">
                  <div className="bg-white rounded-xl shadow-sm hover:shadow-md transition-all duration-200 p-6 border border-gray-100 hover:border-green-200">
                    <div className="flex items-center mb-3">
                      <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center group-hover:bg-green-200 transition-colors">
                        <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-gray-800 ml-3">Train Model</h3>
                    </div>
                    <p className="text-gray-600 text-sm">Configure preprocessing and hyperparameters to start training.</p>
                  </div>
                </a>

                {/* Jobs */}
                <a href="/jobs" className="group">
                  <div className="bg-white rounded-xl shadow-sm hover:shadow-md transition-all duration-200 p-6 border border-gray-100 hover:border-purple-200">
                    <div className="flex items-center mb-3">
                      <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center group-hover:bg-purple-200 transition-colors">
                        <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-gray-800 ml-3">Jobs</h3>
                    </div>
                    <p className="text-gray-600 text-sm">Monitor progress, view logs, and manage your training jobs.</p>
                  </div>
                </a>

                {/* Models */}
                <a href="/models" className="group">
                  <div className="bg-white rounded-xl shadow-sm hover:shadow-md transition-all duration-200 p-6 border border-gray-100 hover:border-rose-200">
                    <div className="flex items-center mb-3">
                      <div className="w-10 h-10 bg-rose-100 rounded-lg flex items-center justify-center group-hover:bg-rose-200 transition-colors">
                        <svg className="w-5 h-5 text-rose-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-gray-800 ml-3">Models</h3>
                    </div>
                    <p className="text-gray-600 text-sm">Browse, evaluate, and export your trained models.</p>
                  </div>
                </a>

                {/* Predictions */}
                <a href="/predict" className="group">
                  <div className="bg-white rounded-xl shadow-sm hover:shadow-md transition-all duration-200 p-6 border border-gray-100 hover:border-teal-200">
                    <div className="flex items-center mb-3">
                      <div className="w-10 h-10 bg-teal-100 rounded-lg flex items-center justify-center group-hover:bg-teal-200 transition-colors">
                        <svg className="w-5 h-5 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-gray-800 ml-3">Make Predictions</h3>
                    </div>
                    <p className="text-gray-600 text-sm">Use deployed models to generate predictions on new data.</p>
                  </div>
                </a>

                {/* Account */}
                <a href="/account" className="group">
                  <div className="bg-white rounded-xl shadow-sm hover:shadow-md transition-all duration-200 p-6 border border-gray-100 hover:border-gray-300">
                    <div className="flex items-center mb-3">
                      <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center group-hover:bg-gray-200 transition-colors">
                        <svg className="w-5 h-5 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5.121 17.804A7 7 0 0112 15a7 7 0 016.879 2.804M15 11a3 3 0 10-6 0 3 3 0 006 0z" />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-gray-800 ml-3">Account</h3>
                    </div>
                    <p className="text-gray-600 text-sm">Manage your profile, API keys, and security settings.</p>
                  </div>
                </a>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <ProtectedRoute>
      <DashboardContent />
    </ProtectedRoute>
  );
}

