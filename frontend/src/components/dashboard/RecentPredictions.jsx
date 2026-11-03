import { useEffect, useState } from 'react'
import { getPredictionHistory } from '../../services/api'
import { Clock, RefreshCw } from 'lucide-react'

// Bright, high-contrast, eye-catching badge styles without blues
const sentimentStyles = {
  'Very Dissatisfied': 'bg-rose-50 text-rose-700 border border-rose-200/80',
  'Dissatisfied': 'bg-orange-50 text-orange-700 border border-orange-200/80',
  'Neutral': 'bg-amber-50 text-amber-700 border border-amber-200/80',
  'Satisfied': 'bg-emerald-50 text-emerald-700 border border-emerald-200/80',
  'Very Satisfied': 'bg-violet-50 text-violet-700 border border-violet-200/80',
}

function RecentPredictions() {
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const fetchHistory = async () => {
    try {
      setError('')
      const data = await getPredictionHistory(5, 0)
      setPredictions(data.predictions)
    } catch (err) {
      console.error(err)
      setError('Unable to load recent predictions.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHistory()
  }, [])

  return (
    <div className="rounded-2xl border border-purple-100/90 bg-white p-6 shadow-xs transition hover:shadow-md">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h3 className="text-lg font-bold text-slate-900 tracking-tight">
            Recent Predictions
          </h3>
          <p className="mt-0.5 text-xs text-slate-500 font-medium">
            Latest customer sentiment analyses processed by the system
          </p>
        </div>

        <button
          onClick={fetchHistory}
          className="rounded-xl border border-purple-200/80 bg-white p-2.5 text-slate-600 shadow-xs transition hover:border-violet-300 hover:bg-purple-50 hover:text-violet-700 active:scale-95"
          aria-label="Refresh predictions"
        >
          <RefreshCw size={16} />
        </button>
      </div>

      {loading ? (
        <div className="py-12 text-center text-sm font-medium text-slate-400">
          Loading recent analyses...
        </div>
      ) : error ? (
        <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-center text-sm font-medium text-rose-600">
          {error}
        </div>
      ) : predictions.length === 0 ? (
        <div className="py-12 text-center text-sm font-medium text-slate-400">
          No predictions recorded yet. Start by analyzing a review!
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-slate-100">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-purple-100 bg-purple-50/50 text-[11px] font-bold uppercase tracking-wider text-purple-950/70">
              <tr>
                <th className="px-5 py-3.5">Review</th>
                <th className="px-5 py-3.5">Sentiment</th>
                <th className="px-5 py-3.5">Confidence</th>
                <th className="px-5 py-3.5">Model</th>
                <th className="px-5 py-3.5">Date</th>
              </tr>
            </thead>

            <tbody className="divide-y divide-slate-100 bg-white">
              {predictions.map((item) => (
                <tr key={item.id} className="transition-colors hover:bg-purple-50/30">
                  <td className="max-w-56 px-5 py-4 font-medium text-slate-800">
                    <span className="block truncate">
                      {item.review_title || item.review_text || 'Image review'}
                    </span>
                  </td>

                  <td className="px-5 py-4">
                    <span className={`whitespace-nowrap rounded-full px-3 py-1 text-xs font-semibold ${sentimentStyles[item.sentiment] || 'bg-slate-100 text-slate-700'}`}>
                      {item.sentiment}
                    </span>
                  </td>

                  <td className="px-5 py-4 font-semibold text-slate-900">
                    {(item.confidence * 100).toFixed(2)}%
                  </td>

                  <td className="px-5 py-4 text-xs font-medium text-slate-500">
                    <span className="rounded-md bg-slate-100 px-2 py-1">
                      {item.model_used.replaceAll('_', ' ')}
                    </span>
                  </td>

                  <td className="whitespace-nowrap px-5 py-4 text-xs text-slate-400 font-medium">
                    <span className="flex items-center gap-1.5">
                      <Clock size={13} className="text-slate-400" />
                      {new Date(item.created_at).toLocaleString()}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export default RecentPredictions
