import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  BrainCircuit,
  History,
  Activity,
  ShieldCheck,
} from 'lucide-react'

const navigation = [
  {
    name: 'Dashboard',
    path: '/',
    icon: LayoutDashboard,
  },
  {
    name: 'Analyze Sentiment',
    path: '/analyze',
    icon: BrainCircuit,
  },
  {
    name: 'Prediction History',
    path: '/history',
    icon: History,
  },
]

function Sidebar() {
  const [isHovered, setIsHovered] = useState(false)

  return (
    <aside
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`fixed left-0 top-0 z-50 flex h-screen flex-col border-r border-purple-900/30 bg-gradient-to-b from-[#1c1137] via-[#140b2a] to-[#0e071e] text-white transition-all duration-300 ease-in-out ${
        isHovered
          ? 'w-64 shadow-2xl shadow-purple-950/60'
          : 'w-20 shadow-lg shadow-purple-950/20'
      }`}
    >
      {/* Brand Header */}
      <div
        className={`flex h-20 items-center border-b border-purple-900/30 overflow-hidden transition-all duration-300 ${
          isHovered ? 'px-4' : 'justify-center px-0'
        }`}
      >
        <div
          className={`flex items-center min-w-max transition-all duration-300 ${
            isHovered ? 'gap-3' : 'justify-center'
          }`}
        >
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-gradient-to-tr from-violet-600 via-purple-600 to-fuchsia-500 text-white shadow-lg shadow-purple-900/50">
            <Activity className="h-6 w-6 text-white" />
          </div>

          <div
            className={`transition-all duration-300 whitespace-nowrap overflow-hidden ${
              isHovered ? 'opacity-100 max-w-[170px]' : 'opacity-0 max-w-0 pointer-events-none'
            }`}
          >
            <div className="flex items-center gap-1.5">
              <h1 className="text-sm font-extrabold tracking-wider bg-gradient-to-r from-white via-purple-100 to-pink-200 bg-clip-text text-transparent">
                SENTIMENT AI
              </h1>
              <span className="flex h-1.5 w-1.5 rounded-full bg-fuchsia-400 animate-pulse" />
            </div>
            <p className="text-[11px] font-medium text-purple-300/70">
              Multimodal Analysis
            </p>
          </div>
        </div>
      </div>

      {/* Navigation List */}
      <div className="flex-1 px-3 py-6 overflow-y-auto overflow-x-hidden">
        <div
          className={`px-2 mb-3 text-[10px] font-bold uppercase tracking-widest text-purple-300/50 transition-all duration-300 whitespace-nowrap overflow-hidden ${
            isHovered ? 'opacity-100 max-w-[180px]' : 'opacity-0 max-w-0 pointer-events-none'
          }`}
        >
          Workspace
        </div>

        <nav className="space-y-2.5">
          {navigation.map((item) => {
            const Icon = item.icon

            return (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === '/'}
                title={!isHovered ? item.name : undefined}
                className={({ isActive }) =>
                  `group relative flex h-12 items-center rounded-xl transition-all duration-200 ${
                    isHovered ? 'px-3.5 gap-3 justify-start' : 'justify-center px-0'
                  } ${
                    isActive
                      ? 'bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white font-semibold shadow-lg shadow-violet-600/35'
                      : 'text-purple-200/75 hover:bg-white/10 hover:text-white'
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <div className="flex shrink-0 items-center justify-center">
                      <Icon
                        size={21}
                        className={`transition-transform duration-200 ${
                          isActive
                            ? 'scale-105 text-white'
                            : 'group-hover:scale-110 text-purple-200/80 group-hover:text-white'
                        }`}
                      />
                    </div>

                    <span
                      className={`text-sm tracking-tight transition-all duration-300 whitespace-nowrap overflow-hidden ${
                        isHovered
                          ? 'opacity-100 max-w-[160px]'
                          : 'opacity-0 max-w-0 pointer-events-none'
                      }`}
                    >
                      {item.name}
                    </span>

                    {/* Left active marker when sidebar is collapsed */}
                    {!isHovered && isActive && (
                      <span className="absolute left-1 top-1/2 -translate-y-1/2 h-6 w-1 rounded-full bg-white shadow-sm" />
                    )}
                  </>
                )}
              </NavLink>
            )
          })}
        </nav>
      </div>

      {/* Footer Info */}
      <div
        className={`border-t border-purple-900/30 overflow-hidden transition-all duration-300 ${
          isHovered ? 'p-4' : 'py-4 flex justify-center px-0'
        }`}
      >
        <div
          className={`flex items-center min-w-max transition-all duration-300 ${
            isHovered ? 'gap-3' : 'justify-center'
          }`}
        >
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-500/10 border border-emerald-500/20 shadow-xs">
            <ShieldCheck className="h-5 w-5 text-emerald-400" />
          </div>

          <div
            className={`transition-all duration-300 whitespace-nowrap overflow-hidden ${
              isHovered ? 'opacity-100 max-w-[170px]' : 'opacity-0 max-w-0 pointer-events-none'
            }`}
          >
            <p className="text-xs font-semibold text-white">
              AI System
            </p>
            <p className="text-[11px] font-medium text-emerald-400 flex items-center gap-1.5">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-emerald-400" />
              Online & Ready
            </p>
          </div>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar
