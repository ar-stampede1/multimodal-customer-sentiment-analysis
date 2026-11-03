import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import Header from './Header'

function Layout() {
  return (
    <div className="min-h-screen bg-[#f8f9fd] text-slate-900 antialiased selection:bg-purple-500 selection:text-white">
      <Sidebar />

      <div className="ml-20 min-h-screen flex flex-col transition-[margin] duration-300">
        <Header />

        <main className="flex-1 p-6 lg:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

export default Layout
