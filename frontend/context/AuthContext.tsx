"use client";

import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { loginUser } from "@/lib/auth";

export type AuthState = {
  token: string | null;
  loading: boolean;
  setToken: (t: string | null) => void;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
};

const AuthContext = createContext<AuthState | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setTokenState] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const t = typeof window !== "undefined" ? localStorage.getItem("auth_token") : null;
    if (t) setTokenState(t);
    setLoading(false);
  }, []);

  const setToken = useCallback((t: string | null) => {
    setTokenState(t);
    if (typeof window !== "undefined") {
      if (t) localStorage.setItem("auth_token", t);
      else localStorage.removeItem("auth_token");
    }
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const res = await loginUser(email, password);
    setToken(res.access_token);
  }, [setToken]);

  const logout = useCallback(() => {
    setToken(null);
  }, [setToken]);

  const value = useMemo(() => ({ token, loading, setToken, login, logout }), [token, loading, setToken, login, logout]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
