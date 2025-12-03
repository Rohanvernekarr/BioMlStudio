import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';

export function Header() {
  const router = useRouter();

  const handleLogout = () => {
    api.clearToken();
    router.push('/login');
  };

  return (
    <header className="border-b border-zinc-800/40 bg-black/95 backdrop-blur-md sticky top-0 z-50 shadow-xl">
      <div className=" mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center gap-8">
            <button 
              onClick={() => router.push('/')}
              className="flex items-center gap-3 hover:opacity-90 transition-all duration-200 group"
            >
              <div className="w-10 h-10 bg-gradient-to-br from-white to-zinc-200 rounded-xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-shadow">
                <span className="text-black font-bold text-base">BM</span>
              </div>
              <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white to-zinc-300 bg-clip-text text-transparent">BioMLStudio</h1>
            </button>
            
            <nav className="hidden md:flex items-center gap-1">
              <button
                onClick={() => router.push('/')}
                className="px-4 py-2 text-sm font-medium text-zinc-400 hover:text-white hover:bg-zinc-900/50 rounded-lg transition-all duration-200"
              >
                🏠 Home
              </button>
            </nav>
          </div>
          
          <button
            onClick={handleLogout}
            className="px-4 py-2 text-sm font-medium text-zinc-300 hover:text-white hover:bg-zinc-900/80 rounded-lg transition-all duration-200 border border-zinc-800/60 hover:border-zinc-700/60"
          >
            🚪 Logout
          </button>
        </div>
      </div>
    </header>
  );
}
