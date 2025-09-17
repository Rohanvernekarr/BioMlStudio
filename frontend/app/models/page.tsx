"use client";

import React, { useState } from "react";
import Link from "next/link";
import { API_BASE_URL } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";

type ModelItem = {
  id: number;
  name: string;
  algorithm?: string;
  model_type: string;
  framework: string;
  created_at: string;
};

type ModelsResponse = {
  models: ModelItem[];
  total: number;
  skip: number;
  limit: number;
};

export default function ModelsListPage() {
  const { token } = useAuth();
  const [models, setModels] = useState<ModelItem[] | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  async function fetchModels() {
    setStatus("Loading models...");
    try {
      const res = await fetch(`${API_BASE_URL}/models`, {
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      });
      if (!res.ok) throw new Error(await res.text());
      const data: ModelsResponse = await res.json();
      setModels(data.models);
      setStatus(`Loaded ${data.models.length} models`);
    } catch (err: any) {
      setStatus(`Failed: ${err.message || String(err)}`);
    }
  }

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-4">
      <h1 className="text-2xl font-semibold">Models</h1>

      <div className="flex items-end gap-3">
        <button onClick={fetchModels} className="bg-blue-600 text-white px-4 py-2 rounded h-10">Load</button>
      </div>

      {status && <p className="text-sm">{status}</p>}

      {models && (
        <div className="overflow-auto border rounded">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="bg-gray-50">
                <th className="text-left p-2">ID</th>
                <th className="text-left p-2">Name</th>
                <th className="text-left p-2">Algorithm</th>
                <th className="text-left p-2">Type</th>
                <th className="text-left p-2">Created</th>
                <th className="text-left p-2">Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.id} className="border-t">
                  <td className="p-2">{m.id}</td>
                  <td className="p-2">{m.name}</td>
                  <td className="p-2">{m.algorithm || "-"}</td>
                  <td className="p-2">{m.model_type}</td>
                  <td className="p-2">{new Date(m.created_at).toLocaleString()}</td>
                  <td className="p-2">
                    <Link className="text-indigo-600 underline" href={`/models/${m.id}/predict`}>Predict</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="text-xs text-gray-500">Set NEXT_PUBLIC_API_BASE_URL to point to your backend API.</p>
    </div>
  );
}
