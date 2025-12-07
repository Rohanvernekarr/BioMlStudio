'use client'

import { useState, useCallback } from 'react'
import { Upload, ArrowDownAz,Dna, Microscope, Target, Shield, Activity, TrendingUp, Download, Play } from 'lucide-react'
import DetailedResults from '@/components/DetailedResults'
import { useAuth } from '@/lib/useAuth'
import { api } from '@/lib/api'

interface DNASequence {
  id: string
  sequence: string
  length: number
  type: string
}

interface AnalysisResult {
  analysis_id: string
  summary: {
    total_sequences: number
    analysis_timestamp: string
    sequence_ids: string[]
  }
  gene_discovery?: any
  mutation_analysis?: any
  drug_targets?: any
  pathogen_detection?: any
  motif_analysis?: any
  biomarker_generation?: any
  evolutionary_analysis?: any
}

const DNADiscoveryPage = () => {
  const { isAuthenticated } = useAuth()
  const [sequences, setSequences] = useState<DNASequence[]>([])
  const [inputSequence, setInputSequence] = useState('')
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedAnalyses, setSelectedAnalyses] = useState({
    gene_discovery: true,
    mutation_analysis: true,
    drug_targets: true,
    pathogen_detection: true,
    motif_analysis: true,
    biomarker_generation: true,
    evolutionary_analysis: true
  })

  const handleAddSequence = () => {
    if (inputSequence.trim()) {
      const newSequence: DNASequence = {
        id: `seq_${Date.now()}`,
        sequence: inputSequence.trim().toUpperCase(),
        length: inputSequence.trim().length,
        type: detectSequenceType(inputSequence.trim())
      }
      setSequences([...sequences, newSequence])
      setInputSequence('')
    }
  }

  const detectSequenceType = (seq: string): string => {
    const upperSeq = seq.toUpperCase()
    const nucleotides = new Set(upperSeq)
    
    if (nucleotides.size <= 4 && Array.from(nucleotides).every(n => 'ATCG'.includes(n))) {
      return 'DNA'
    } else if (nucleotides.size <= 4 && Array.from(nucleotides).every(n => 'AUCG'.includes(n))) {
      return 'RNA'
    } else {
      return 'PROTEIN'
    }
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    try {
      const text = await file.text()
      const fastaSequences = parseFASTA(text)
      setSequences([...sequences, ...fastaSequences])
    } catch (error) {
      console.error('Error reading file:', error)
      alert('Error reading file. Please ensure it is a valid FASTA format.')
    }
  }

  const parseFASTA = (text: string): DNASequence[] => {
    const sequences: DNASequence[] = []
    const lines = text.split('\n')
    let currentId = ''
    let currentSequence = ''

    lines.forEach(line => {
      line = line.trim()
      if (line.startsWith('>')) {
        if (currentId && currentSequence) {
          sequences.push({
            id: currentId,
            sequence: currentSequence.toUpperCase(),
            length: currentSequence.length,
            type: detectSequenceType(currentSequence)
          })
        }
        currentId = line.substring(1).split(' ')[0]
        currentSequence = ''
      } else if (line) {
        currentSequence += line.replace(/[^ATCGURYNWSMKBDHV]/gi, '')
      }
    })

    if (currentId && currentSequence) {
      sequences.push({
        id: currentId,
        sequence: currentSequence.toUpperCase(),
        length: currentSequence.length,
        type: detectSequenceType(currentSequence)
      })
    }

    return sequences
  }

  const filteredSequences = sequences.filter(seq => 
    seq.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    seq.sequence.toLowerCase().includes(searchTerm.toLowerCase()) ||
    seq.type.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const runAnalysis = async () => {
    if (sequences.length === 0) {
      alert('Please add sequences first')
      return
    }

    // Estimate processing time based on sequence count and length
    const totalBasePairs = sequences.reduce((sum, seq) => sum + seq.length, 0)
    const estimatedMinutes = Math.ceil((totalBasePairs / 10000) + (sequences.length * 0.1))
    
    const confirmAnalysis = confirm(
      `Starting analysis of ${sequences.length} sequences (${totalBasePairs.toLocaleString()} bp total).\\n` +
      `Estimated processing time: ${estimatedMinutes} minute(s).\\n\\n` +
      `Continue with analysis?`
    )
    
    if (!confirmAnalysis) return

    setLoading(true)
    try {
      const results = await api.analyzeDNA({
        sequences: sequences.map(s => s.sequence),
        sequence_ids: sequences.map(s => s.id),
        analysis_config: selectedAnalyses
      })
      setAnalysisResults(results)
    } catch (error) {
      console.error('Analysis error:', error)
      alert('Analysis failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const downloadResults = () => {
    if (!analysisResults) return

    const dataStr = JSON.stringify(analysisResults, null, 2)
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr)
    
    const exportFileDefaultName = `dna_analysis_${analysisResults.analysis_id}.json`
    
    const linkElement = document.createElement('a')
    linkElement.setAttribute('href', dataUri)
    linkElement.setAttribute('download', exportFileDefaultName)
    linkElement.click()
  }

  return (
    <div className="min-h-screen bg-zinc-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">
            DNA Discovery Studio
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Advanced AI-powered bioinformatics analysis for genomic datasets. 
            Extract biological insights from DNA sequences for drug discovery, 
            pathogen detection, disease prediction, and genetic classification.
          </p>
        </div>

        {/* Input Section */}
        <div className="bg-zinc-900 rounded-lg shadow-lg p-6 mb-8 border border-zinc-700">
          <h2 className="text-2xl font-semibold mb-4 flex items-center text-white">
            <Dna className="mr-2 text-blue-400" />
            Sequence Input
          </h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Manual Input */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Enter DNA Sequence
              </label>
              <textarea
                value={inputSequence}
                onChange={(e) => setInputSequence(e.target.value)}
                placeholder="Paste your DNA sequence here (ATCG format)..."
                className="w-full h-32 p-3 bg-gray-700 border border-gray-600 rounded-md focus:border-transparent text-white placeholder-gray-400"
              />
              <button
                onClick={handleAddSequence}
                className="mt-2 px-4 py-2 bg-white text-black rounded-md transition-colors"
              >
                Add Sequence
              </button>
            </div>

            {/* File Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Upload FASTA File
              </label>
              <div className="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center bg-gray-750 hover:border-gray-500 transition-colors">
                <Upload className="mx-auto h-12 w-12 text-gray-400" />
                <div className="mt-2">
                  <label htmlFor="file-upload" className="cursor-pointer">
                    <span className="text-blue-400 hover:text-blue-300 transition-colors">
                      Upload a file
                    </span>
                    <input
                      id="file-upload"
                      name="file-upload"
                      type="file"
                      accept=".fasta,.fa,.fas,.txt"
                      className="sr-only"
                      onChange={handleFileUpload}
                    />
                  </label>
                </div>
                <p className="text-sm text-gray-400">FASTA format supported</p>
              </div>
            </div>
          </div>

          {/* Current Sequences */}
          {sequences.length > 0 && (
            <div className="mt-6">
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-lg font-medium text-white">
                  Loaded Sequences ({sequences.length})
                </h3>
                <button
                  onClick={() => setSequences([])}
                  className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700 transition-colors"
                >
                  Clear All
                </button>
              </div>
              <div className="max-h-80 overflow-y-auto space-y-2 scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800">
                {sequences.slice(0, 100).map((seq, index) => (
                  <div key={seq.id} className="border border-gray-600 rounded-lg p-4 bg-zinc-700 hover:bg-gray-650 transition-colors">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-medium text-sm truncate text-white">{seq.id}</h4>
                      <span className="text-xs bg-zinc-700 text-blue-100 px-2 py-1 rounded">
                        {seq.type}
                      </span>
                    </div>
                    <p className="text-xs text-gray-300 mb-2">
                      Length: {seq.length.toLocaleString()} bp
                    </p>
                    <p className="text-xs font-mono bg-zinc-900 text-gray-200 p-2 rounded border border-gray-500 truncate">
                      {seq.sequence.substring(0, 50)}
                      {seq.sequence.length > 50 ? '...' : ''}
                    </p>
                    <button
                      onClick={() => setSequences(sequences.filter((_, i) => i !== index))}
                      className="mt-2 text-red-600 hover:text-red-800 text-xs"
                    >
                      Remove
                    </button>
                  </div>
                ))}
                {sequences.length > 100 && (
                  <div className="p-3 bg-yellow-900 bg-opacity-50 border border-yellow-700 rounded-lg text-center">
                    <span className="text-yellow-300 text-sm">
                      📊 Displaying first 100 of {sequences.length} sequences for performance. All {sequences.length} sequences will be included in analysis.
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Analysis Configuration */}
        <div className="bg-zinc-800 rounded-lg shadow-lg p-6 mb-8 border border-gray-700">
          <h2 className="text-2xl font-semibold mb-4 text-white">Analysis Configuration</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { key: 'gene_discovery', label: 'Gene Discovery', icon: Dna, description: 'Identify new genes and ORFs' },
              { key: 'mutation_analysis', label: 'Mutation Analysis', icon: Target, description: 'Find disease-causing mutations' },
              { key: 'drug_targets', label: 'Drug Targets', icon: Target, description: 'Identify potential drug targets' },
              { key: 'pathogen_detection', label: 'Pathogen Detection', icon: Shield, description: 'Detect pathogens and resistance' },
              { key: 'motif_analysis', label: 'Motif Analysis', icon: Activity, description: 'Find functional motifs' },
              { key: 'biomarker_generation', label: 'Biomarkers', icon: TrendingUp, description: 'Generate diagnostic biomarkers' },
              { key: 'evolutionary_analysis', label: 'Evolution', icon: Microscope, description: 'Evolutionary analysis' }
            ].map(({ key, label, icon: Icon, description }) => (
              <div key={key} className="border border-gray-600 rounded-lg p-4 bg-gray-700 hover:bg-gray-650 transition-colors">
                <div className="flex items-center mb-2">
                  <input
                    type="checkbox"
                    id={key}
                    checked={selectedAnalyses[key as keyof typeof selectedAnalyses]}
                    onChange={(e) => setSelectedAnalyses({
                      ...selectedAnalyses,
                      [key]: e.target.checked
                    })}
                    className="mr-2 w-4 h-4 text-blue-600 bg-gray-600 border-gray-500 rounded focus:ring-2"
                  />
                  <Icon className="h-5 w-5 text-blue-400 mr-2" />
                  <label htmlFor={key} className="font-medium text-sm text-white cursor-pointer">
                    {label}
                  </label>
                </div>
                <p className="text-xs text-gray-300">{description}</p>
              </div>
            ))}
          </div>
          
          <div className="mt-6">
            <button
              onClick={runAnalysis}
              disabled={loading || sequences.length === 0}
              className="px-8 py-3 bg-white text-black rounded-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center text-lg font-medium transition-all transform hover:scale-105 active:scale-95"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-5 w-5" />
                  Run Analysis
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results Section */}
        {analysisResults && (
          <div className="mb-8">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-semibold text-white">Analysis Results</h2>
              <button
                onClick={downloadResults}
                className="px-4 py-2 bg-white text-black rounded-lg flex items-center transition-colors"
              >
                <Download className="mr-2 h-4 w-4" />
                Download Results
              </button>
            </div>
            
            <DetailedResults analysisResults={analysisResults} />
          </div>
        )}

      </div>
    </div>
  )
}

export default DNADiscoveryPage