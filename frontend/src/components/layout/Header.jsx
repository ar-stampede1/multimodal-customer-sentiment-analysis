import { Bell, Sparkles, Activity } from 'lucide-react'

function Header() {
  return (
    <header className="sticky top-0 z-30 flex h-20 items-center justify-between border-b border-purple-100/80 bg-white/85 px-6 lg:px-8 backdrop-blur-md shadow-xs">
      <div>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 rounded-full bg-violet-100 px-2.5 py-0.5 text-xs font-semibold text-violet-700">
            <Sparkles size={12} className="text-violet-600" />
            AI Intelligence
          </span>
          <span className="text-xs text-slate-400 hidden sm:inline">
            Customer Sentiment Platform
          </span>
        </div>
        <h2 className="mt-0.5 text-lg font-bold text-slate-900 tracking-tight">
          Multimodal Sentiment Analysis
        </h2>
      </div>

      <div className="flex items-center gap-3 sm:gap-4">
        <button
          className="relative rounded-xl border border-purple-100 bg-white p-2.5 text-slate-600 shadow-xs transition hover:border-violet-300 hover:bg-purple-50 hover:text-violet-700"
          aria-label="Notifications"
        >
          <Bell size={18} />
          <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-fuchsia-500 ring-2 ring-white" />
        </button>

        <div className="flex items-center gap-2.5 rounded-full border border-purple-100 bg-white py-1.5 pl-1.5 pr-3.5 shadow-xs">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-tr from-violet-600 to-fuchsia-500 text-xs font-bold text-white shadow-sm shadow-violet-500/30">
            AI
          </div>
          <div className="hidden sm:block text-left">
            <p className="text-xs font-semibold text-slate-800 leading-tight">Admin System</p>
            <p className="text-[10px] text-emerald-600 font-medium flex items-center gap-1 leading-tight">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-emerald-500" />
              Connected
            </p>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
