import { InputHTMLAttributes } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

export function Input({ label, error, className = '', ...props }: InputProps) {
  return (
    <div className="flex flex-col gap-1.5">
      {label && <label className="text-sm text-zinc-400">{label}</label>}
      <input
        className={`bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-2 text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-white ${className}`}
        {...props}
      />
      {error && <span className="text-sm text-red-400">{error}</span>}
    </div>
  );
}
