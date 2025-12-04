import { InputHTMLAttributes } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

export function Input({ label, error, className = '', ...props }: InputProps) {
  return (
    <div className="flex flex-col gap-3">
      {label && (
        <label className="text-sm font-semibold text-zinc-200 tracking-wide">
          {label}
        </label>
      )}
      <input
        className={`bg-gradient-to-br from-zinc-900/90 to-zinc-950/90 border border-zinc-800/60 rounded-xl px-5 py-4 text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-zinc-500/20 focus:border-zinc-600/60 hover:border-zinc-700/60 transition-all duration-200 backdrop-blur-sm shadow-lg hover:shadow-xl min-h-[52px] ${
          error ? 'border-red-500/60 focus:ring-red-500/20' : ''
        } ${className}`}
        {...props}
      />
      {error && (
        <span className="text-sm text-red-400 font-medium flex items-center gap-2">
          <span className="text-red-500">⚠️</span>
          {error}
        </span>
      )}
    </div>
  );
}
