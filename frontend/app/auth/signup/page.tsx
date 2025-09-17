"use client";

import React, { useState } from "react";
import Link from "next/link";
import { registerUser } from "@/lib/auth";
import { useAuth } from "@/context/AuthContext";

export default function SignupPage() {
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [status, setStatus] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("Creating your account...");
    try {
      await registerUser({ email, password, full_name: fullName || undefined });
      // Auto login after successful registration
      await login(email, password);
      setStatus("Signup successful. You are now logged in.");
    } catch (err: any) {
      setStatus(`Signup failed: ${err.message || String(err)}`);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-6">
      <div className="w-full max-w-md bg-white rounded-xl shadow p-6 space-y-4">
        <h1 className="text-2xl font-semibold text-center">Create an account</h1>
        <form onSubmit={onSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium">Full name</label>
            <input
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              className="mt-1 w-full border rounded px-3 py-2"
              placeholder="Jane Doe"
            />
          </div>
          <div>
            <label className="block text-sm font-medium">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 w-full border rounded px-3 py-2"
              placeholder="jane@example.com"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 w-full border rounded px-3 py-2"
              placeholder="••••••••"
              required
            />
          </div>
          <button type="submit" className="w-full bg-blue-600 text-white px-4 py-2 rounded">
            Sign up
          </button>
        </form>

        {status && <p className="text-sm text-center">{status}</p>}

        <p className="text-center text-sm text-gray-600">
          Already have an account? {" "}
          <Link href="/auth/login" className="text-blue-600 underline">
            Log in
          </Link>
        </p>
      </div>
    </div>
  );
}
