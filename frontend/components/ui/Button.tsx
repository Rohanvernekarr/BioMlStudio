import { ButtonHTMLAttributes, ReactNode } from 'react';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  children: ReactNode;
}

export function Button({ 
  variant = 'primary', 
  size = 'md', 
  className = '', 
  children, 
  ...props 
}: ButtonProps) {
  const baseStyles = 'inline-flex items-center justify-center rounded-xl font-semibold transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-black disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.98] select-none';
  
  const variants = {
    primary: 'bg-gradient-to-r from-white to-zinc-100 text-black hover:from-zinc-100 hover:to-zinc-200 shadow-lg hover:shadow-xl focus:ring-white/20 border border-transparent',
    secondary: 'bg-gradient-to-r from-zinc-800 to-zinc-900 text-white hover:from-zinc-700 hover:to-zinc-800 border border-zinc-700 hover:border-zinc-600 shadow-md hover:shadow-lg focus:ring-zinc-500/20',
    ghost: 'bg-transparent text-zinc-200 hover:bg-zinc-900/80 hover:text-white focus:ring-zinc-500/20 border border-transparent',
    outline: 'bg-transparent text-zinc-200 hover:bg-zinc-900/50 hover:text-white border border-zinc-700 hover:border-zinc-500 focus:ring-zinc-500/20',
  };

  const sizes = {
    sm: 'px-4 py-2.5 text-sm min-h-[36px]',
    md: 'px-6 py-3 text-base min-h-[44px]',
    lg: 'px-8 py-4 text-lg min-h-[52px]',
  };

  return (
    <button
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
