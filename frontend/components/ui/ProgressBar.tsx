interface ProgressBarProps {
  value: number;
  max?: number;
  className?: string;
}

export function ProgressBar({ value, max = 100, className = '' }: ProgressBarProps) {
  const percentage = Math.min((value / max) * 100, 100);

  return (
    <div className={`w-full bg-zinc-800 rounded-full h-2 ${className}`}>
      <div
        className="bg-white h-2 rounded-full transition-all duration-500"
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
}
