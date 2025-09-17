"use client";

import React, { useState } from "react";
import { API_BASE_URL } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";

export default function UploadDatasetPage() {
  const { token } = useAuth();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [datasetType, setDatasetType] = useState("general");
  const [isPublic, setIsPublic] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("Uploading...");
    setResult(null);

    if (!file) {
      setStatus("Please choose a file");
      return;
    }

    const form = new FormData();
    form.append("file", file);
    form.append("name", name);
    form.append("description", description);
    form.append("dataset_type", datasetType);
    form.append("is_public", String(isPublic));

    try {
      const res = await fetch(`${API_BASE_URL}/datasets/upload`, {
        method: "POST",
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
        body: form,
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setResult(data);
      setStatus("Upload successful");
    } catch (err: any) {
      setStatus(`Upload failed: ${err.message || String(err)}`);
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

      {status && <p className="mt-4 text-sm">{status}</p>}

      {result && (
        <pre className="mt-4 p-3 bg-gray-100 rounded text-sm overflow-auto">
{JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  );
}
