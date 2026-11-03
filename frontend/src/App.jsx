
import { BrowserRouter, Routes, Route } from 'react-router-dom'

import Layout from './components/layout/Layout'

import Dashboard from './pages/Dashboard'
import Analyze from './pages/Analyze'
import History from './pages/History'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/history" element={<History />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
