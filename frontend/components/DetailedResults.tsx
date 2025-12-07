'use client'

import { useState } from 'react'
import { ChevronDown, ChevronRight, Info, ExternalLink, Copy, CheckCircle } from 'lucide-react'
import BioVisualization from './BioVisualization'

interface DetailedResultsProps {
  analysisResults: any
}

const DetailedResults: React.FC<DetailedResultsProps> = ({ analysisResults }) => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['summary']))
  const [copiedText, setCopiedText] = useState<string>('')

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections)
    if (newExpanded.has(section)) {
      newExpanded.delete(section)
    } else {
      newExpanded.add(section)
    }
    setExpandedSections(newExpanded)
  }

  const copyToClipboard = (text: string, type: string) => {
    navigator.clipboard.writeText(text)
    setCopiedText(type)
    setTimeout(() => setCopiedText(''), 2000)
  }

  const formatSequence = (sequence: string, maxLength: number = 50) => {
    if (sequence.length <= maxLength) return sequence
    return sequence.substring(0, maxLength) + '...'
  }

  const BiologicalInsightCard = ({ title, insight, confidence, relevance }: any) => (
    <div className="border-l-4 border-blue-500 pl-4 py-3 bg-blue-50 rounded-r-lg">
      <h4 className="font-semibold text-blue-800">{title}</h4>
      <p className="text-sm text-gray-700 mt-1">{insight}</p>
      <div className="flex justify-between mt-2 text-xs">
        <span className="text-blue-600">Confidence: {confidence}%</span>
        <span className="text-gray-600">Clinical Relevance: {relevance}</span>
      </div>
    </div>
  )

  const sections = [
    {
      key: 'gene_discovery',
      title: 'Gene Discovery & ORF Analysis',
      icon: '🧬',
      description: 'Newly identified genes and their coding potential'
    },
    {
      key: 'mutation_analysis',
      title: 'Disease Mutation Analysis',
      icon: '⚠️',
      description: 'Disease-causing mutations and oncogenic patterns'
    },
    {
      key: 'drug_targets',
      title: 'Drug Target Identification',
      icon: '🎯',
      description: 'Potential therapeutic targets and binding sites'
    },
    {
      key: 'pathogen_detection',
      title: 'Pathogen & Resistance Detection',
      icon: '🦠',
      description: 'Infectious agents and antibiotic resistance'
    },
    {
      key: 'motif_analysis',
      title: 'Functional Motif Analysis',
      icon: '🔍',
      description: 'Regulatory elements and functional sequences'
    },
    {
      key: 'biomarker_generation',
      title: 'Biomarker Discovery',
      icon: '📊',
      description: 'Diagnostic signatures and disease markers'
    },
    {
      key: 'evolutionary_analysis',
      title: 'Evolutionary Analysis',
      icon: '🌲',
      description: 'Phylogenetic signals and evolutionary pressure'
    }
  ]

  return (
    <div className="space-y-6">
      
      {/* Executive Summary */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6 border border-blue-200">
        <h2 className="text-2xl font-bold text-blue-900 mb-4">🧬 Biological Discovery Summary</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="text-2xl font-bold text-green-600">
              {analysisResults.gene_discovery?.potential_genes?.length || 0}
            </div>
            <div className="text-sm text-gray-600">New Genes Discovered</div>
          </div>
          
          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="text-2xl font-bold text-red-600">
              {analysisResults.mutation_analysis?.statistics?.oncogenic_sites || 0}
            </div>
            <div className="text-sm text-gray-600">Oncogenic Sites</div>
          </div>
          
          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="text-2xl font-bold text-purple-600">
              {analysisResults.drug_targets?.druggable_proteins?.length || 0}
            </div>
            <div className="text-sm text-gray-600">Drug Targets</div>
          </div>
          
          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="text-2xl font-bold text-orange-600">
              {analysisResults.pathogen_detection?.resistance_genes?.length || 0}
            </div>
            <div className="text-sm text-gray-600">Resistance Genes</div>
          </div>
        </div>

        {/* Key Biological Insights */}
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-blue-800">🔬 Key Biological Insights</h3>
          
          {analysisResults.gene_discovery?.potential_genes?.length > 0 && (
            <BiologicalInsightCard
              title="Novel Gene Discovery"
              insight={`Identified ${analysisResults.gene_discovery.potential_genes.length} potential new genes with significant coding potential. These sequences may represent previously uncharacterized protein-coding regions with potential therapeutic or diagnostic value.`}
              confidence={85}
              relevance="High"
            />
          )}
          
          {analysisResults.mutation_analysis?.statistics?.oncogenic_sites > 0 && (
            <BiologicalInsightCard
              title="Cancer-Associated Mutations"
              insight={`Detected ${analysisResults.mutation_analysis.statistics.oncogenic_sites} oncogenic mutation sites that may contribute to cancer development. These variants require further investigation for clinical significance.`}
              confidence={92}
              relevance="Critical"
            />
          )}
          
          {analysisResults.drug_targets?.druggable_proteins?.length > 0 && (
            <BiologicalInsightCard
              title="Therapeutic Target Potential"
              insight={`Found ${analysisResults.drug_targets.druggable_proteins.length} proteins with high druggability scores. These targets present opportunities for drug development and therapeutic intervention.`}
              confidence={78}
              relevance="High"
            />
          )}
          
          {analysisResults.pathogen_detection?.resistance_genes?.length > 0 && (
            <BiologicalInsightCard
              title="Antimicrobial Resistance"
              insight={`Identified ${analysisResults.pathogen_detection.resistance_genes.length} antibiotic resistance genes. This suggests potential challenges for antimicrobial therapy and requires resistance profiling.`}
              confidence={88}
              relevance="Critical"
            />
          )}
        </div>
      </div>

      {/* Detailed Analysis Sections */}
      {sections.map((section) => {
        const sectionData = analysisResults[section.key]
        if (!sectionData) return null

        const isExpanded = expandedSections.has(section.key)

        return (
          <div key={section.key} className="bg-white rounded-lg shadow-lg border border-gray-200">
            <div
              className="flex items-center justify-between p-6 cursor-pointer hover:bg-gray-50"
              onClick={() => toggleSection(section.key)}
            >
              <div className="flex items-center space-x-3">
                <span className="text-2xl">{section.icon}</span>
                <div>
                  <h3 className="text-xl font-semibold text-gray-900">{section.title}</h3>
                  <p className="text-sm text-gray-600">{section.description}</p>
                </div>
              </div>
              {isExpanded ? <ChevronDown className="h-5 w-5" /> : <ChevronRight className="h-5 w-5" />}
            </div>

            {isExpanded && (
              <div className="border-t border-gray-200 p-6">
                
                {/* Visualization */}
                <div className="mb-6">
                  <h4 className="text-lg font-semibold mb-3">📈 Data Visualization</h4>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <BioVisualization 
                      data={sectionData} 
                      type={section.key as any}
                    />
                  </div>
                </div>

                {/* Detailed Results */}
                <div className="space-y-4">
                  <h4 className="text-lg font-semibold">🔬 Detailed Findings</h4>
                  
                  {section.key === 'gene_discovery' && sectionData.potential_genes && (
                    <div className="space-y-3">
                      {sectionData.potential_genes.slice(0, 5).map((gene: any, idx: number) => (
                        <div key={idx} className="border rounded-lg p-4 bg-green-50">
                          <div className="flex justify-between items-start mb-2">
                            <h5 className="font-medium">Gene Candidate #{idx + 1}</h5>
                            <span className="text-xs bg-green-200 text-green-800 px-2 py-1 rounded">
                              Coding Potential: {(gene.coding_potential * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <span className="text-gray-600">Frame:</span> {gene.frame}
                            </div>
                            <div>
                              <span className="text-gray-600">Length:</span> {gene.length} bp
                            </div>
                            <div>
                              <span className="text-gray-600">Start:</span> {gene.start}
                            </div>
                            <div>
                              <span className="text-gray-600">End:</span> {gene.end}
                            </div>
                          </div>
                          <div className="mt-2">
                            <span className="text-gray-600">Protein Sequence:</span>
                            <div className="flex items-center space-x-2 mt-1">
                              <code className="text-xs bg-white p-2 rounded border font-mono">
                                {formatSequence(gene.protein_seq)}
                              </code>
                              <button
                                onClick={() => copyToClipboard(gene.protein_seq, `gene-${idx}`)}
                                className="p-1 text-gray-400 hover:text-gray-600"
                              >
                                {copiedText === `gene-${idx}` ? <CheckCircle className="h-4 w-4 text-green-500" /> : <Copy className="h-4 w-4" />}
                              </button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {section.key === 'mutation_analysis' && (
                    <div className="space-y-3">
                      {sectionData.oncogenic_patterns?.slice(0, 5).map((pattern: any, idx: number) => (
                        <div key={idx} className="border rounded-lg p-4 bg-red-50">
                          <div className="flex justify-between items-start mb-2">
                            <h5 className="font-medium text-red-800">{pattern.motif_name}</h5>
                            <span className="text-xs bg-red-200 text-red-800 px-2 py-1 rounded">
                              Risk: {(pattern.oncogenic_risk * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="text-sm space-y-1">
                            <div><span className="text-gray-600">Position:</span> {pattern.position}</div>
                            <div><span className="text-gray-600">Sequence:</span> <code className="bg-white px-2 py-1 rounded">{pattern.sequence}</code></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {section.key === 'drug_targets' && (
                    <div className="space-y-3">
                      {sectionData.druggable_proteins?.slice(0, 5).map((protein: any, idx: number) => (
                        <div key={idx} className="border rounded-lg p-4 bg-purple-50">
                          <div className="flex justify-between items-start mb-2">
                            <h5 className="font-medium">Drug Target #{idx + 1}</h5>
                            <span className="text-xs bg-purple-200 text-purple-800 px-2 py-1 rounded">
                              Druggability: {(protein.druggability_score * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="text-sm space-y-1">
                            <div><span className="text-gray-600">Frame:</span> {protein.frame}</div>
                            <div>
                              <span className="text-gray-600">Protein:</span>
                              <code className="block bg-white p-2 rounded mt-1 text-xs font-mono">
                                {formatSequence(protein.protein, 80)}
                              </code>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {section.key === 'pathogen_detection' && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h5 className="font-medium mb-2">🦠 Pathogen Signatures</h5>
                        {(sectionData.bacterial_signatures || []).concat(sectionData.viral_signatures || []).slice(0, 3).map((sig: any, idx: number) => (
                          <div key={idx} className="border rounded p-3 mb-2 bg-orange-50">
                            <div className="text-sm">
                              <div><strong>{sig.signature_type}</strong></div>
                              <div>Organism: {sig.organism_type}</div>
                              <div>Confidence: {(sig.confidence * 100).toFixed(0)}%</div>
                            </div>
                          </div>
                        ))}
                      </div>
                      <div>
                        <h5 className="font-medium mb-2">💊 Resistance Genes</h5>
                        {(sectionData.resistance_genes || []).slice(0, 3).map((gene: any, idx: number) => (
                          <div key={idx} className="border rounded p-3 mb-2 bg-red-50">
                            <div className="text-sm">
                              <div><strong>{gene.resistance_type}</strong></div>
                              <div>Antibiotic Class: {gene.antibiotic_class}</div>
                              <div>Frame: {gene.frame}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {section.key === 'motif_analysis' && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h5 className="font-medium mb-2">🔰 Regulatory Elements</h5>
                        {(sectionData.promoters || []).concat(sectionData.enhancers || []).slice(0, 3).map((motif: any, idx: number) => (
                          <div key={idx} className="border rounded p-3 mb-2 bg-indigo-50">
                            <div className="text-sm">
                              <div><strong>{motif.promoter_type || motif.enhancer_type}</strong></div>
                              <div>Position: {motif.position}</div>
                              <div>Sequence: <code>{motif.sequence}</code></div>
                            </div>
                          </div>
                        ))}
                      </div>
                      <div>
                        <h5 className="font-medium mb-2">🧬 CpG Islands</h5>
                        {(sectionData.cpg_islands || []).slice(0, 3).map((island: any, idx: number) => (
                          <div key={idx} className="border rounded p-3 mb-2 bg-teal-50">
                            <div className="text-sm">
                              <div>Region: {island.start} - {island.end}</div>
                              <div>GC Content: {(island.gc_content * 100).toFixed(1)}%</div>
                              <div>CpG Ratio: {island.cpg_ratio.toFixed(2)}%</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {section.key === 'biomarker_generation' && (
                    <div className="space-y-3">
                      <h5 className="font-medium">🧬 Discriminative K-mers</h5>
                      {(sectionData.discriminative_kmers || []).slice(0, 5).map((kmer: any, idx: number) => (
                        <div key={idx} className="border rounded-lg p-4 bg-cyan-50">
                          <div className="flex justify-between items-start mb-2">
                            <code className="font-bold text-lg">{kmer.kmer}</code>
                            <span className="text-xs bg-cyan-200 text-cyan-800 px-2 py-1 rounded">
                              {kmer.fold_change.toFixed(1)}x enriched
                            </span>
                          </div>
                          <div className="text-sm space-y-1">
                            <div><span className="text-gray-600">Associated with:</span> {kmer.associated_label}</div>
                            <div><span className="text-gray-600">Frequency in group:</span> {(kmer.frequency_in_group * 100).toFixed(2)}%</div>
                            <div><span className="text-gray-600">Frequency in others:</span> {(kmer.frequency_in_others * 100).toFixed(2)}%</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                </div>

                {/* Clinical Relevance */}
                <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                  <h4 className="text-lg font-semibold text-blue-800 mb-2">🏥 Clinical Relevance</h4>
                  <div className="text-sm text-gray-700">
                    {section.key === 'gene_discovery' && (
                      <p>Novel gene discoveries may lead to new therapeutic targets and biomarkers. Validation through functional studies is recommended.</p>
                    )}
                    {section.key === 'mutation_analysis' && (
                      <p>Oncogenic mutations require clinical correlation and may influence treatment decisions. Consider genetic counseling for patients.</p>
                    )}
                    {section.key === 'drug_targets' && (
                      <p>Identified drug targets present opportunities for precision medicine approaches. Further structural and functional validation needed.</p>
                    )}
                    {section.key === 'pathogen_detection' && (
                      <p>Pathogen signatures and resistance genes inform infection control and antimicrobial selection strategies.</p>
                    )}
                    {section.key === 'motif_analysis' && (
                      <p>Regulatory motifs provide insights into gene expression control and potential epigenetic modifications.</p>
                    )}
                    {section.key === 'biomarker_generation' && (
                      <p>Discriminative biomarkers can be developed into diagnostic assays with appropriate validation studies.</p>
                    )}
                  </div>
                </div>

              </div>
            )}
          </div>
        )
      })}

      {/* Research Recommendations */}
      <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-6 border border-green-200">
        <h2 className="text-xl font-bold text-green-900 mb-4">🎯 Research & Development Recommendations</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-green-800 mb-3">Immediate Actions</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start">
                <span className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                <span>Validate top gene candidates through RT-PCR or sequencing</span>
              </li>
              <li className="flex items-start">
                <span className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                <span>Cross-reference mutations with clinical variant databases</span>
              </li>
              <li className="flex items-start">
                <span className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                <span>Perform structural modeling of identified drug targets</span>
              </li>
              <li className="flex items-start">
                <span className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                <span>Conduct antimicrobial susceptibility testing</span>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-green-800 mb-3">Long-term Studies</h3>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start">
                <span className="w-2 h-2 bg-emerald-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                <span>Functional characterization of novel genes</span>
              </li>
              <li className="flex items-start">
                <span className="w-2 h-2 bg-emerald-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                <span>Population-scale validation of biomarkers</span>
              </li>
              <li className="flex items-start">
                <span className="w-2 h-2 bg-emerald-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                <span>Drug screening against identified targets</span>
              </li>
              <li className="flex items-start">
                <span className="w-2 h-2 bg-emerald-500 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                <span>Longitudinal studies for disease progression</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

    </div>
  )
}

export default DetailedResults