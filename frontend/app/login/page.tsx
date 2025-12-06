'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { api } from '@/lib/api';
import { AlertTriangle, Loader2, LogIn, Sparkles } from 'lucide-react';

export default function LoginPage() {
  const router = useRouter();
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      if (isLogin) {
        await api.login(email, password);
        router.push('/');
      } else {
        await api.register(email, password, fullName);
        await api.login(email, password);
        router.push('/');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Authentication failed');
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-zinc-950 to-black flex-center-col px-4 sm:px-6 lg:px-8 padding-section">
      <div className="centered-card">
        <div className="text-center-wrapper section-spacing">
          <div className="inline-flex items-center gap-3 mb-8">
            <div className="w-16 h-16 bg-gradient-to-br from-white to-zinc-300 rounded-2xl flex items-center justify-center shadow-2xl">
              <span className="text-black font-bold text-2xl">BM</span>
            </div>
          </div>
          <h1 className="text-5xl sm:text-6xl font-bold mb-6 tracking-tight bg-gradient-to-br from-white via-zinc-100 to-zinc-400 bg-clip-text text-transparent">
            BioMLStudio
          </h1>
          <p className="text-zinc-400 text-xl font-light">
            {isLogin ? 'Welcome back' : 'Create your account'}
          </p>
        </div>

        <Card className="!p-10 lg:!p-12 card-spacing">
          <form onSubmit={handleSubmit} className="gap-component flex flex-col">
            {!isLogin && (
              <Input
                label="Full Name"
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="John Doe"
                required
              />
            )}

            <Input
              label="Email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
            />

            <Input
              label="Password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
            />

            {error && (
              <div className="p-6 bg-red-950/40 border border-red-800/60 rounded-xl text-red-300 text-sm font-medium flex items-center gap-2">
                <AlertTriangle className="w-5 h-5" />
                {error}
              </div>
            )}

            <Button type="submit" disabled={loading} className="w-full !py-4 !text-lg font-semibold mt-2 flex items-center justify-center gap-2" size="lg">
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Please wait...
                </>
              ) : isLogin ? (
                <>
                  <LogIn className="w-5 h-5" />
                  Sign In
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  Create Account
                </>
              )}
            </Button>
          </form>

          <div className="mt-8 pt-8 border-t border-zinc-800 text-center element-spacing">
            <button
              type="button"
              onClick={() => {
                setIsLogin(!isLogin);
                setError('');
              }}
              className="text-sm text-zinc-400 hover:text-white transition-colors"
            >
              {isLogin ? "Don't have an account? Sign up" : 'Already have an account? Sign in'}
            </button>
          </div>
        </Card>
      </div>
    </div>
  );
}
