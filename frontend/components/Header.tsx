import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';

export function Header() {
  const router = useRouter();

  const handleLogout = () => {
    api.clearToken();
    router.push('/login');
  };

  return (
    <header className="border-b border-zinc-800 bg-black">
      <div className="max-w-7xl mx-auto px-8 py-4 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <h1 className="text-xl font-bold">BioMLStudio</h1>
        </div>
        <button
          onClick={handleLogout}
          className="text-sm text-zinc-400 hover:text-white transition-colors"
        >
          Logout
        </button>
      </div>
    </header>
  );
}
