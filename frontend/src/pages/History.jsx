import { useEffect, useState } from 'react'
import {
  Search,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  Clock,
  Database,
  Filter,
} from 'lucide-react'

import { getPredictionHistory } from '../services/api'

// High-contrast, bright, eye-catching badge styles without blues
const sentimentStyles = {
  'Very Dissatisfied': 'bg-rose-50 text-rose-700 border-rose-200',
  'Dissatisfied': 'bg-orange-50 text-orange-700 border-orange-200',
  'Neutral': 'bg-amber-50 text-amber-700 border-amber-200',
  'Satisfied': 'bg-emerald-50 text-emerald-700 border-emerald-200',
  'Very Satisfied': 'bg-violet-50 text-violet-700 border-violet-200',
}

function History() {
  const [predictions, setPredictions] = useState([])
  const [total, setTotal] = useState(0)
  const [limit] = useState(10)
  const [offset, setOffset] = useState(0)

  const [search, setSearch] = useState('')
  const [sentimentFilter, setSentimentFilter] = useState('All')
  const [modelFilter, setModelFilter] = useState('All')

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const fetchHistory = async () => {
    setLoading(true)
    setError('')

    try {
      const data = await getPredictionHistory(limit, offset)
      setPredictions(data.predictions)
      setTotal(data.total)
    } catch (err) {
      console.error('History API error:', err)
      setError('Unable to retrieve prediction history.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHistory()
  }, [offset])

  const filteredPredictions = predictions.filter((item) => {
    const searchMatch = [
      item.review_title,
      item.review_text,
      item.sentiment,
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase()
      .includes(search.toLowerCase())

    const sentimentMatch =
      sentimentFilter === 'All' ||
      item.sentiment === sentimentFilter

    const modelMatch =
      modelFilter === 'All' ||
      item.model_used === modelFilter

    return searchMatch && sentimentMatch && modelMatch
  })

  const totalPages = Math.ceil(total / limit)
  const currentPage = Math.floor(offset / limit) + 1

  const formatDate = (date) => {
    return new Date(date).toLocaleString()
  }

  return (
    <div className="space-y-8">
      {/* Heading */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-900 tracking-tight">
            Prediction History
          </h1>

          <p className="mt-1 text-sm text-slate-500">
            View, filter, and inspect previously analyzed customer reviews and sentiment records.
          </p>
        </div>

        <button
          onClick={fetchHistory}
          disabled={loading}
          className="flex items-center gap-2 rounded-xl border border-purple-200/80 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 shadow-xs transition hover:border-violet-300 hover:bg-purple-50 hover:text-violet-700 active:scale-95 disabled:opacity-50"
        >
          <RefreshCw size={16} />
          Refresh Records
        </button>
      </div>

      {/* Summary Banner */}
      <div className="rounded-2xl border border-purple-100/90 bg-white p-6 shadow-xs">
        <div className="flex items-center gap-4">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-violet-100 text-violet-700 shadow-xs">
            <Database size={26} />
          </div>

          <div>
            <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Total Recorded Predictions
            </p>

            <h2 className="mt-0.5 text-3xl font-extrabold text-slate-900 tracking-tight">
              {total}
            </h2>
          </div>
        </div>
      </div>

      {/* Search and Filters */}
      <div className="rounded-2xl border border-purple-100/90 bg-white p-6 shadow-xs">
        <div className="mb-4 flex items-center gap-2">
          <Filter size={18} className="text-violet-600" />
          <h3 className="text-base font-bold text-slate-900 tracking-tight">
            Search and Filter Records
          </h3>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div className="relative">
            <Search
              size={17}
              className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400"
            />

            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by keywords or sentiment..."
              className="w-full rounded-xl border border-slate-200 bg-slate-50/60 py-3 pl-10 pr-4 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:bg-white focus:border-violet-500 focus:ring-4 focus:ring-violet-500/10"
            />
          </div>

          <select
            value={sentimentFilter}
            onChange={(e) => setSentimentFilter(e.target.value)}
            className="rounded-xl border border-slate-200 bg-slate-50/60 px-4 py-3 text-sm font-medium text-slate-700 outline-none transition focus:bg-white focus:border-violet-500 focus:ring-4 focus:ring-violet-500/10"
          >
            <option value="All">All Sentiments</option>
            {Object.keys(sentimentStyles).map((sentiment) => (
              <option key={sentiment} value={sentiment}>
                {sentiment}
              </option>
            ))}
          </select>

          <select
            value={modelFilter}
            onChange={(e) => setModelFilter(e.target.value)}
            className="rounded-xl border border-slate-200 bg-slate-50/60 px-4 py-3 text-sm font-medium text-slate-700 outline-none transition focus:bg-white focus:border-violet-500 focus:ring-4 focus:ring-violet-500/10"
          >
            <option value="All">All Model Architectures</option>
            <option value="text_only">Text Only</option>
            <option value="image_only">Image Only</option>
            <option value="multimodal">Multimodal</option>
          </select>
        </div>
      </div>

      {/* Prediction Table */}
      <div className="overflow-hidden rounded-2xl border border-purple-100/90 bg-white shadow-xs">
        <div className="border-b border-purple-100/80 px-6 py-4">
          <h3 className="text-base font-bold text-slate-900 tracking-tight">
            Recorded Sentiment Predictions
          </h3>

          <p className="mt-0.5 text-xs text-slate-500 font-medium">
            Showing {filteredPredictions.length} of {total} records on this page
          </p>
        </div>

        {loading ? (
          <div className="py-16 text-center text-sm font-medium text-slate-400">
            Loading prediction history...
          </div>
        ) : error ? (
          <div className="p-8 text-center text-rose-600">
            <p className="font-semibold">{error}</p>
            <button
              onClick={fetchHistory}
              className="mt-3 rounded-lg bg-rose-50 px-4 py-1.5 text-xs font-bold text-rose-700 border border-rose-200 hover:bg-rose-100"
            >
              Retry
            </button>
          </div>
        ) : filteredPredictions.length === 0 ? (
          <div className="py-16 text-center text-sm font-medium text-slate-400">
            No matching predictions found. Try adjusting your search query or filters.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-purple-100 bg-purple-50/50 text-[11px] font-bold uppercase tracking-wider text-purple-950/70">
                <tr>
                  <th className="px-6 py-4">ID</th>
                  <th className="px-6 py-4">Review Content</th>
                  <th className="px-6 py-4">Sentiment</th>
                  <th className="px-6 py-4">Confidence</th>
                  <th className="px-6 py-4">Model</th>
                  <th className="px-6 py-4">Image</th>
                  <th className="px-6 py-4">Date</th>
                </tr>
              </thead>

              <tbody className="divide-y divide-slate-100 bg-white">
                {filteredPredictions.map((item) => (
                  <tr
                    key={item.id}
                    className="transition hover:bg-purple-50/30"
                  >
                    <td className="px-6 py-4 font-bold text-slate-400 text-xs">
                      #{item.id}
                    </td>

                    <td className="max-w-xs px-6 py-4 font-medium text-slate-800">
                      <span className="block truncate">
                        {item.review_title ||
                          item.review_text ||
                          'Image review'}
                      </span>
                    </td>

                    <td className="px-6 py-4">
                      <span
                        className={`whitespace-nowrap rounded-full border px-3 py-1 text-xs font-semibold ${sentimentStyles[item.sentiment] || 'bg-slate-100 text-slate-700'}`}
                      >
                        {item.sentiment}
                      </span>
                    </td>

                    <td className="px-6 py-4 font-bold text-slate-900">
                      {(item.confidence * 100).toFixed(2)}%
                    </td>

                    <td className="px-6 py-4 text-xs font-medium text-slate-600">
                      <span className="rounded-md bg-slate-100 px-2 py-1">
                        {item.model_used.replace('_', ' ')}
                      </span>
                    </td>

                    <td className="px-6 py-4 text-xs font-semibold text-slate-600">
                      {item.image_used ? (
                        <span className="text-emerald-600">Yes</span>
                      ) : (
                        <span className="text-slate-400">No</span>
                      )}
                    </td>

                    <td className="whitespace-nowrap px-6 py-4 text-xs font-medium text-slate-400">
                      <span className="flex items-center gap-1.5">
                        <Clock size={13} />
                        {formatDate(item.created_at)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination */}
        {!loading && !error && totalPages > 0 && (
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-purple-100/80 px-6 py-4 bg-slate-50/40">
            <p className="text-xs font-medium text-slate-500">
              Page {currentPage} of {totalPages}
            </p>

            <div className="flex gap-2">
              <button
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - limit))}
                className="flex items-center gap-1 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-bold text-slate-700 shadow-xs transition hover:border-violet-300 hover:bg-purple-50 hover:text-violet-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                <ChevronLeft size={15} />
                Previous
              </button>

              <button
                disabled={offset + limit >= total}
                onClick={() => setOffset(offset + limit)}
                className="flex items-center gap-1 rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-bold text-slate-700 shadow-xs transition hover:border-violet-300 hover:bg-purple-50 hover:text-violet-700 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Next
                <ChevronRight size={15} />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default History
