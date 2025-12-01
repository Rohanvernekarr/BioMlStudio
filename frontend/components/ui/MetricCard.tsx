interface MetricCardProps {
  label: string;
  value: string | number;
  good?: boolean;
}

export function MetricCard({ label, value, good }: MetricCardProps) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
      <div className="text-sm text-zinc-400 mb-2">{label}</div>
      <div className={`text-3xl font-bold ${good ? 'text-green-400' : 'text-white'}`}>
        {typeof value === 'number' ? value.toFixed(3) : value}
      </div>
    </div>
  );
}
