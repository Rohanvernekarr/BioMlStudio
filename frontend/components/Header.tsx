import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';

export function Header() {
  const router = useRouter();

  const handleLogout = () => {
    api.clearToken();
    router.push('/login');
  };

  return (
    <header className="border-b border-zinc-800/50 bg-black/95 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 sm:px-8 lg:px-12 py-5 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-white rounded-lg flex items-center justify-center">
            <span className="text-black font-bold text-sm">BM</span>
          </div>
          <h1 className="text-xl font-bold tracking-tight">BioMLStudio</h1>
        </div>
        <button
          onClick={handleLogout}
          className="px-4 py-2 text-sm text-zinc-400 hover:text-white hover:bg-zinc-900 rounded-lg transition-all duration-200"
        >
          Logout
        </button>
      </div>
    </header>
  );
}
