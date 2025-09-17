"use client";

import React, { useMemo, useState } from "react";
import { API_BASE_URL } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";

type ModelType = "classification" | "regression";

export default function CreateJobPage() {
  const { token } = useAuth();
  const [name, setName] = useState("Training Job");
  const [description, setDescription] = useState("");
  const [modelType, setModelType] = useState<ModelType>("classification");
  const [datasetPath, setDatasetPath] = useState("");
  const [algorithm, setAlgorithm] = useState("random_forest");
  const [testSize, setTestSize] = useState(0.2);
  const [randomState, setRandomState] = useState(42);
  const [scale, setScale] = useState(true);

  // Hyperparameters
  const [nEstimators, setNEstimators] = useState(100);
  const [maxDepth, setMaxDepth] = useState<number | "">("");
  const [minSamplesSplit, setMinSamplesSplit] = useState(2);
  const [minSamplesLeaf, setMinSamplesLeaf] = useState(1);

  const [status, setStatus] = useState<string | null>(null);
  const [createdJob, setCreatedJob] = useState<any>(null);

  const hyperparams = useMemo(() => {
    const hp: Record<string, any> = {};
    if (algorithm === "random_forest") {
      hp.n_estimators = nEstimators;
      if (maxDepth !== "" && maxDepth !== null) hp.max_depth = Number(maxDepth);
      hp.min_samples_split = minSamplesSplit;
      hp.min_samples_leaf = minSamplesLeaf;
      hp.random_state = randomState;
    }
    if (algorithm === "logistic_regression") {
      hp.max_iter = 1000;
      hp.random_state = randomState;
    }
    if (algorithm === "linear_regression") {
      // no specific hyperparams required
    }
    return hp;
  }, [algorithm, nEstimators, maxDepth, minSamplesSplit, minSamplesLeaf, randomState]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("Creating job...");
    setCreatedJob(null);

    if (!datasetPath) {
      setStatus("Please provide a dataset path on the server");
      return;
    }

    const payload = {
      name,
      description,
      job_type: "training",
      config: {
        model_type: modelType,
        dataset_path: datasetPath,
        test_size: testSize,
        random_state: randomState,
        scale_features: scale,
        algorithm,
        hyperparameters: hyperparams,
      },
    };

    try {
      const res = await fetch(`${API_BASE_URL}/jobs`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setCreatedJob(data);
      setStatus("Job created. Background training started.");
    } catch (err: any) {
      setStatus(`Failed: ${err.message || String(err)}`);
    }
  }

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-4">
      <h1 className="text-2xl font-semibold">Create Training Job</h1>
      <p className="text-sm text-gray-600">This will POST to /jobs with job_type = training and your configuration.</p>

      <form onSubmit={onSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium">Job Name</label>
            <input
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="mt-1 w-full border rounded px-3 py-2"
            />
          </div>
          <div>
            <label className="block text-sm font-medium">Model Type</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value as ModelType)}
              className="mt-1 w-full border rounded px-3 py-2"
            >
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>
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
          <label className="block text-sm font-medium">Dataset Path (server file path)</label>
          <input
            required
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.target.value)}
            placeholder="e.g., /app/data/user123/dataset.csv"
            className="mt-1 w-full border rounded px-3 py-2"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium">Algorithm</label>
            <select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
              className="mt-1 w-full border rounded px-3 py-2"
            >
              <option value="random_forest">Random Forest</option>
              {modelType === "classification" && (
                <option value="logistic_regression">Logistic Regression</option>
              )}
              {modelType === "regression" && (
                <option value="linear_regression">Linear Regression</option>
              )}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium">Test Size</label>
            <input
              type="number"
              min={0.05}
              max={0.9}
              step={0.05}
              value={testSize}
              onChange={(e) => setTestSize(Number(e.target.value))}
              className="mt-1 w-full border rounded px-3 py-2"
            />
          </div>
          <div>
            <label className="block text-sm font-medium">Random State</label>
            <input
              type="number"
              value={randomState}
              onChange={(e) => setRandomState(parseInt(e.target.value, 10))}
              className="mt-1 w-full border rounded px-3 py-2"
            />
          </div>
        </div>

        {algorithm === "random_forest" && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium">n_estimators</label>
              <input
                type="number"
                value={nEstimators}
                onChange={(e) => setNEstimators(parseInt(e.target.value, 10))}
                className="mt-1 w-full border rounded px-3 py-2"
              />
            </div>
            <div>
              <label className="block text-sm font-medium">max_depth</label>
              <input
                type="number"
                value={maxDepth === "" ? "" : maxDepth}
                onChange={(e) => setMaxDepth(e.target.value === "" ? "" : parseInt(e.target.value, 10))}
                placeholder="empty = none"
                className="mt-1 w-full border rounded px-3 py-2"
              />
            </div>
            <div>
              <label className="block text-sm font-medium">min_samples_split</label>
              <input
                type="number"
                value={minSamplesSplit}
                onChange={(e) => setMinSamplesSplit(parseInt(e.target.value, 10))}
                className="mt-1 w-full border rounded px-3 py-2"
              />
            </div>
            <div>
              <label className="block text-sm font-medium">min_samples_leaf</label>
              <input
                type="number"
                value={minSamplesLeaf}
                onChange={(e) => setMinSamplesLeaf(parseInt(e.target.value, 10))}
                className="mt-1 w-full border rounded px-3 py-2"
              />
            </div>
          </div>
        )}

        <div className="flex items-center space-x-2">
          <input id="scale" type="checkbox" checked={scale} onChange={(e) => setScale(e.target.checked)} />
          <label htmlFor="scale">Scale features</label>
        </div>

        <button type="submit" className="bg-green-600 text-white px-4 py-2 rounded">
          Create Job
        </button>
      </form>

      {status && <p className="text-sm mt-3">{status}</p>}

      {createdJob && (
        <pre className="mt-4 p-3 bg-gray-100 rounded text-sm overflow-auto">{JSON.stringify(createdJob, null, 2)}</pre>
      )}
    </div>
  );
}
