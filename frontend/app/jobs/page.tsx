"use client";

import React, { useState } from "react";
import { API_BASE_URL } from "@/lib/api";
import { useAuth } from "@/context/AuthContext";

type JobItem = {
  id: number;
  name: string;
  job_type: string;
  status: string;
  progress_percent: number;
  current_step?: string | null;
  created_at: string;
  updated_at?: string | null;
  model_id?: number | null;
};

type JobsResponse = {
  jobs: JobItem[];
  total: number;
  skip: number;
  limit: number;
};

export default function JobsListPage() {
  const { token } = useAuth();
  const [jobs, setJobs] = useState<JobItem[] | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  async function fetchJobs() {
    setStatus("Loading jobs...");
    try {
      const res = await fetch(`${API_BASE_URL}/jobs`, {
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      });
      if (!res.ok) throw new Error(await res.text());
      const data: JobsResponse = await res.json();
      setJobs(data.jobs);
      setStatus(`Loaded ${data.jobs.length} jobs`);
    } catch (err: any) {
      setStatus(`Failed: ${err.message || String(err)}`);
    }
  }

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-4">
      <h1 className="text-2xl font-semibold">Jobs</h1>

      <div className="flex items-end gap-3">
        <button onClick={fetchJobs} className="bg-blue-600 text-white px-4 py-2 rounded h-10">Load</button>
      </div>

      {status && <p className="text-sm">{status}</p>}

      {jobs && (
        <div className="overflow-auto border rounded">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="bg-gray-50">
                <th className="text-left p-2">ID</th>
                <th className="text-left p-2">Name</th>
                <th className="text-left p-2">Type</th>
                <th className="text-left p-2">Status</th>
                <th className="text-left p-2">Progress</th>
                <th className="text-left p-2">Current Step</th>
                <th className="text-left p-2">Model</th>
                <th className="text-left p-2">Created</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id} className="border-t">
                  <td className="p-2">{j.id}</td>
                  <td className="p-2">{j.name}</td>
                  <td className="p-2">{j.job_type}</td>
                  <td className="p-2">{j.status}</td>
                  <td className="p-2">{j.progress_percent?.toFixed(1)}%</td>
                  <td className="p-2">{j.current_step || "-"}</td>
                  <td className="p-2">{j.model_id ?? "-"}</td>
                  <td className="p-2">{new Date(j.created_at).toLocaleString()}</td>
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
