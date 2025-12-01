import { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
}

export function Card({ children, className = '' }: CardProps) {
  return (
    <div className={`bg-zinc-900 border border-zinc-800 rounded-lg p-6 ${className}`}>
      {children}
    </div>
  );
}
