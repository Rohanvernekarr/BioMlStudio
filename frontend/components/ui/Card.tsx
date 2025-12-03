import { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
}

export function Card({ children, className = '' }: CardProps) {
  return (
    <div className={`bg-zinc-900/50 border border-zinc-800 rounded-xl p-8 shadow-2xl backdrop-blur-sm ${className}`}>
      {children}
    </div>
  );
}
