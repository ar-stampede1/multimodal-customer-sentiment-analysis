import { useEffect, useState } from 'react'
import RecentPredictions from '../components/dashboard/RecentPredictions'

import {
  Database,
  Target,
  FileText,
  Layers,
  RefreshCw,
  Sparkles,
} from 'lucide-react'

import { getDashboardStatistics } from '../services/api'

import StatCard from '../components/dashboard/StatCard'
import SentimentChart from '../components/dashboard/SentimentChart'
import ModelUsageChart from '../components/dashboard/ModelUsageChart'

function Dashboard() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const fetchStatistics = async () => {
    try {
      setError('')
      const data = await getDashboardStatistics()
      setStats(data)
    } catch (err) {
      console.error(err)
      setError('Unable to load dashboard statistics.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStatistics()
  }, [])

  if (loading) {
    return (
      <div className="flex min-h-[300px] items-center justify-center">
        <div className="flex items-center gap-3 rounded-2xl border border-purple-100 bg-white px-6 py-4 shadow-xs">
          <RefreshCw className="h-5 w-5 animate-spin text-violet-600" />
          <p className="text-sm font-medium text-slate-600">Loading dashboard metrics...</p>
        </div>
      </div>
    )
  }

  if (error || !stats) {
    return (
      <div className="flex items-center justify-between rounded-2xl border border-rose-200 bg-rose-50/80 p-5 text-rose-700 shadow-xs">
        <span className="text-sm font-medium">{error || 'No dashboard data available.'}</span>
        <button
          onClick={fetchStatistics}
          className="rounded-xl bg-rose-600 px-4 py-1.5 text-xs font-semibold text-white transition hover:bg-rose-500 shadow-xs"
        >
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
            Dashboard Overview
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            Real-time monitor for customer sentiment distribution and multimodal model throughput.
          </p>
        </div>

        <button
          onClick={fetchStatistics}
          className="flex items-center gap-2 rounded-xl border border-purple-200/80 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 shadow-xs transition hover:border-violet-300 hover:bg-purple-50 hover:text-violet-700 active:scale-95"
        >
          <RefreshCw size={16} />
          Refresh Data
        </button>
      </div>

      {/* KPI Stats Grid */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          title="Total Predictions"
          value={stats.total_predictions}
          subtitle="All recorded analyses"
          icon={Database}
          color="bg-violet-100 text-violet-700"
        />

        <StatCard
          title="Average Confidence"
          value={`${(stats.average_confidence * 100).toFixed(2)}%`}
          subtitle="Average model confidence"
          icon={Target}
          color="bg-emerald-100 text-emerald-700"
        />

        <StatCard
          title="Text-only Analysis"
          value={stats.model_usage.text_only}
          subtitle="Text-based predictions"
          icon={FileText}
          color="bg-fuchsia-100 text-fuchsia-700"
        />

        <StatCard
          title="Multimodal Analysis"
          value={stats.model_usage.multimodal}
          subtitle="Text + image predictions"
          icon={Layers}
          color="bg-amber-100 text-amber-700"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <SentimentChart
          data={stats.sentiment_distribution.counts}
        />

        <ModelUsageChart
          data={stats.model_usage}
        />
      </div>

      {/* Recent Activity Table */}
      <RecentPredictions />
    </div>
  )
}

export default Dashboard
