"use client";

import React, { useState } from "react";
import { useParams } from "next/navigation";
import { API_BASE_URL } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";

export default function PredictWithModelPage() {
  // modelId from route
  const params = useParams();
  const modelId = params?.modelId as string;

  const { token } = useAuth();
  const [inputJson, setInputJson] = useState('[{"feature1": 0.5, "feature2": 1.2}]');
  const [returnProb, setReturnProb] = useState(false);

  const [status, setStatus] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("Predicting...");
    setResult(null);

    let inputData: any;
    try {
      inputData = JSON.parse(inputJson);
      if (!Array.isArray(inputData)) throw new Error("input_data must be an array of objects");
    } catch (err: any) {
      setStatus(`Invalid JSON: ${err.message || String(err)}`);
      return;
    }

    const body = {
      input_data: inputData,
      return_probabilities: returnProb,
    };

    try {
      const res = await fetch(`${API_BASE_URL}/models/${modelId}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setResult(data);
      setStatus("Success");
    } catch (err: any) {
      setStatus(`Failed: ${err.message || String(err)}`);
    }
  }

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-4">
      <h1 className="text-2xl font-semibold">Predict with Model #{modelId}</h1>

      <form onSubmit={onSubmit} className="space-y-4">
        <div className="flex items-center space-x-2">
          <input id="returnProb" type="checkbox" checked={returnProb} onChange={(e) => setReturnProb(e.target.checked)} />
          <label htmlFor="returnProb">Return probabilities (if supported)</label>
        </div>

        <div>
          <label className="block text-sm font-medium">Input JSON (array of objects)</label>
          <textarea
            rows={8}
            value={inputJson}
            onChange={(e) => setInputJson(e.target.value)}
            className="mt-1 w-full border rounded px-3 py-2 font-mono text-sm"
          />
        </div>

        <button type="submit" className="bg-indigo-600 text-white px-4 py-2 rounded">Predict</button>
      </form>

      {status && <p className="text-sm mt-3">{status}</p>}

      {result && (
        <pre className="mt-4 p-3 bg-gray-100 rounded text-sm overflow-auto">{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
}
