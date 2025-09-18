"use client";

import React, { useState } from "react";
import { datasetAPI } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";

export default function UploadDatasetPage() {
  const { token } = useAuth();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [datasetType, setDatasetType] = useState("general");
  const [isPublic, setIsPublic] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [result, setResult] = useState<{
    upload?: any;
    preprocessing?: any;
    biological_features?: any;
    training?: any;
  } | null>(null);
  const [processingStep, setProcessingStep] = useState<string>("");
  const [autoPreprocess, setAutoPreprocess] = useState(true);
  const [extractBioFeatures, setExtractBioFeatures] = useState(true);
  const [autoTrain, setAutoTrain] = useState(false);
  const [selectedModel, setSelectedModel] = useState("random_forest");
  const [taskType, setTaskType] = useState("classification");

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("Uploading...");
    setResult(null);

    if (!file) {
      setStatus("Please choose a file");
      return;
    }

    if (!token) {
      setStatus("Authentication required. Please log in again.");
      return;
    }

    console.log("Token available:", !!token);
    console.log("Token length:", token?.length);

    const form = new FormData();
    form.append("file", file);
    form.append("name", name);
    form.append("description", description);
    form.append("dataset_type", datasetType);
    form.append("is_public", String(isPublic));

    try {
      // Step 1: Upload dataset
      setProcessingStep("Uploading dataset...");
      const uploadData = await datasetAPI.uploadDataset(form, token);
      const datasetId = uploadData.id;
      
      setStatus("Upload successful! Starting processing...");
      setResult({ upload: uploadData });

      // Step 2: Auto preprocessing (if enabled)
      if (autoPreprocess) {
        setProcessingStep("Getting preprocessing suggestions...");
        setStatus("Analyzing data and getting preprocessing suggestions...");
        
        try {
          const suggestions = await datasetAPI.getPreprocessingSuggestions(datasetId, "classification", token);
          
          if (suggestions.recommended_steps && suggestions.recommended_steps.length > 0) {
            setProcessingStep("Applying preprocessing steps...");
            setStatus("Applying recommended preprocessing steps...");
            
            const preprocessingConfig = {
              steps: suggestions.recommended_steps.map((step: any) => ({
                name: step.name,
                parameters: step.parameters,
                enabled: true
              }))
            };
            
            const preprocessResult = await datasetAPI.preprocessDataset(datasetId, preprocessingConfig, token);
            setResult(prev => ({ ...prev, preprocessing: preprocessResult }));
          }
        } catch (preprocessErr) {
          console.warn("Preprocessing failed:", preprocessErr);
          setStatus("Upload successful, but preprocessing failed. Continuing with feature extraction...");
        }
      }

      // Step 3: Extract biological features (if enabled and appropriate dataset type)
      if (extractBioFeatures && datasetType !== "general") {
        setProcessingStep("Extracting biological features...");
        setStatus("Extracting specialized biological features...");
        
        try {
          const bioConfig = {
            sequence_column: "sequence",
            sequence_type: datasetType,
            extract_composition: true,
            extract_physicochemical: true,
            extract_kmers: datasetType !== "protein",
            kmer_size: datasetType === "dna" ? 3 : 4
          };
          
          const bioResult = await datasetAPI.extractBiologicalFeatures(datasetId, bioConfig, token);
          setResult(prev => ({ ...prev, biological_features: bioResult }));
        } catch (bioErr) {
          console.warn("Biological feature extraction failed:", bioErr);
          setStatus("Upload and preprocessing complete, but biological feature extraction failed.");
        }
      }

      // Step 4: Start model training (if enabled)
      if (autoTrain) {
        setProcessingStep("Starting model training...");
        setStatus("🤖 Starting model training...");
        
        try {
          const trainingConfig = {
            model_type: selectedModel,
            task_type: taskType,
            dataset_id: datasetId,
            config: {
              test_size: 0.2,
              random_state: 42,
              cross_validation: true,
              cv_folds: 5
            }
          };
          
          const trainingResult = await datasetAPI.startTraining(trainingConfig, token);
          setResult(prev => ({ ...prev, training: trainingResult }));
          setStatus("🚀 Model training started successfully!");
        } catch (trainErr) {
          console.warn("Model training failed:", trainErr);
          setStatus("Dataset processed successfully, but model training failed.");
        }
      }

      // Final status
      setProcessingStep("Complete!");
      if (!autoTrain) {
        setStatus("✅ Dataset uploaded and processed successfully!");
      }
      
    } catch (err: any) {
      console.error("Upload error:", err);
      setStatus(`Upload failed: ${err.message || String(err)}`);
      setProcessingStep("");
    }
  }

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h1 className="text-2xl font-semibold mb-4">Upload Dataset</h1>

      <form onSubmit={onSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium">Name</label>
          <input
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="mt-1 w-full border rounded px-3 py-2"
          />
        </div>

        <div>
          <label className="block text-sm font-medium">Description</label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="mt-1 w-full border rounded px-3 py-2"
          />
        </div>

        <div>
          <label className="block text-sm font-medium">Dataset Type</label>
          <select
            value={datasetType}
            onChange={(e) => setDatasetType(e.target.value)}
            className="mt-1 w-full border rounded px-3 py-2"
          >
            <option value="general">General</option>
            <option value="dna">DNA</option>
            <option value="rna">RNA</option>
            <option value="protein">Protein</option>
          </select>
        </div>

        <div className="flex items-center space-x-2">
          <input
            id="isPublic"
            type="checkbox"
            checked={isPublic}
            onChange={(e) => setIsPublic(e.target.checked)}
          />
          <label htmlFor="isPublic">Public</label>
        </div>

        <div className="flex items-center space-x-2">
          <input
            id="autoPreprocess"
            type="checkbox"
            checked={autoPreprocess}
            onChange={(e) => setAutoPreprocess(e.target.checked)}
          />
          <label htmlFor="autoPreprocess">Auto-preprocess data</label>
        </div>

        <div className="flex items-center space-x-2">
          <input
            id="extractBioFeatures"
            type="checkbox"
            checked={extractBioFeatures}
            onChange={(e) => setExtractBioFeatures(e.target.checked)}
            disabled={datasetType === "general"}
          />
          <label htmlFor="extractBioFeatures" className={datasetType === "general" ? "text-gray-400" : ""}>
            Extract biological features {datasetType === "general" ? "(not available for general datasets)" : ""}
          </label>
        </div>

        <div className="flex items-center space-x-2">
          <input
            id="autoTrain"
            type="checkbox"
            checked={autoTrain}
            onChange={(e) => setAutoTrain(e.target.checked)}
          />
          <label htmlFor="autoTrain">🤖 Auto-train model after processing</label>
        </div>

        {autoTrain && (
          <div className="ml-6 space-y-3 p-4 bg-gray-50 rounded-lg border">
            <div>
              <label className="block text-sm font-medium">Task Type</label>
              <select
                value={taskType}
                onChange={(e) => setTaskType(e.target.value)}
                className="mt-1 w-full border rounded px-3 py-2"
              >
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium">Model Algorithm</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="mt-1 w-full border rounded px-3 py-2"
              >
                <option value="random_forest">Random Forest</option>
                <option value="svm">Support Vector Machine</option>
                <option value="logistic_regression">Logistic Regression</option>
                <option value="gradient_boosting">Gradient Boosting</option>
                <option value="neural_network">Neural Network</option>
              </select>
            </div>
          </div>
        )}

        <div>
          <label className="block text-sm font-medium">File</label>
          <input
            type="file"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="mt-1 w-full"
            accept=".csv,.tsv,.fasta,.fa"
          />
        </div>

        <button
          type="submit"
          className="bg-blue-600 text-white px-4 py-2 rounded"
        >
          Upload
        </button>
      </form>

      {/* Processing Status */}
      {processingStep && (
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
            <span className="text-blue-800 font-medium">{processingStep}</span>
          </div>
        </div>
      )}

      {status && (
        <div className={`mt-4 p-3 rounded-lg text-sm ${
          status.includes("✅") || status.includes("successful") 
            ? "bg-green-50 border border-green-200 text-green-800" 
            : status.includes("failed") || status.includes("error")
            ? "bg-red-50 border border-red-200 text-red-800"
            : "bg-gray-50 border border-gray-200 text-gray-800"
        }`}>
          {status}
        </div>
      )}

      {/* Results Summary */}
      {result && (
        <div className="mt-6 space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">Processing Results</h3>
          
          {result.upload && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <h4 className="font-medium text-green-900 mb-2">✅ Dataset Upload</h4>
              <p className="text-green-800 text-sm">
                Dataset "{result.upload.name}" uploaded successfully (ID: {result.upload.id})
              </p>
            </div>
          )}

          {result.preprocessing && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h4 className="font-medium text-blue-900 mb-2">🔄 Preprocessing</h4>
              <p className="text-blue-800 text-sm">
                Preprocessing job started (Job ID: {result.preprocessing.job_id})
              </p>
            </div>
          )}

          {result.biological_features && (
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <h4 className="font-medium text-purple-900 mb-2">🧬 Biological Features</h4>
              <p className="text-purple-800 text-sm">
                Feature extraction job started (Job ID: {result.biological_features.job_id})
              </p>
            </div>
          )}

          {result.training && (
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
              <h4 className="font-medium text-orange-900 mb-2">🤖 Model Training</h4>
              <p className="text-orange-800 text-sm">
                {selectedModel} training started (Job ID: {result.training.id})
              </p>
              <p className="text-orange-700 text-xs mt-1">
                Task: {taskType} | Algorithm: {selectedModel}
              </p>
            </div>
          )}

          {/* Raw JSON for debugging */}
          <details className="mt-4">
            <summary className="cursor-pointer text-sm text-gray-600 hover:text-gray-800">
              View raw results (for debugging)
            </summary>
            <pre className="mt-2 p-3 bg-gray-800 rounded text-xs overflow-auto max-h-64">
{JSON.stringify(result, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  );
}
