import { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
}

export function Card({ children, className = '' }: CardProps) {
  return (
    <div className={`bg-gradient-to-br from-zinc-900/90 to-zinc-950/90 border border-zinc-800/60 rounded-2xl p-6 sm:p-8 shadow-2xl backdrop-blur-md hover:shadow-3xl hover:border-zinc-700/60 transition-all duration-300 ${className}`}>
      {children}
    </div>
  );
}
