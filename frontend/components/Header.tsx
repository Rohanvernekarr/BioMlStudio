import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';

export function Header() {
  const router = useRouter();

  const handleLogout = () => {
    api.clearToken();
    router.push('/login');
  };

  return (
    <header className="border-b  border-zinc-800/50 items-center justify-between bg-black/95 backdrop-blur-md sticky top-0 z-50 shadow-lg w-full">
      <div className="w-full mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <div className="flex items-center gap-8">
          <button 
            onClick={() => router.push('/')}
            className="flex items-center gap-3 hover:opacity-80 transition-opacity"
          >
            <div className="w-10 h-10 bg-gradient-to-br from-white to-zinc-300 rounded-lg flex items-center justify-center shadow-lg">
              <span className="text-black font-bold text-base">BM</span>
            </div>
            <h1 className="text-xl font-bold tracking-tight">BioMLStudio</h1>
          </button>
          
          <nav className="hidden md:flex items-center gap-6">
            <button
              onClick={() => router.push('/')}
              className="text-sm font-medium text-zinc-400 hover:text-white transition-colors"
            >
              🏠 Home
            </button>
          </nav>
        </div>
        
        <button
          onClick={handleLogout}
          className="px-5 py-2.5 text-sm font-medium text-zinc-300 hover:text-white hover:bg-zinc-900 rounded-lg transition-all duration-200 border border-zinc-800 hover:border-zinc-700"
        >
          🚪 Logout
        </button>
      </div>
    </header>
  );
}
