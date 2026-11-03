import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts'

// High-contrast, vibrant palette without dull blues or cyans
const COLORS = [
  '#f43f5e', // Very Dissatisfied: Vivid Rose Red
  '#fb923c', // Dissatisfied: Vibrant Coral Orange
  '#f59e0b', // Neutral: Bright Warm Amber
  '#10b981', // Satisfied: Fresh Mint Emerald
  '#8b5cf6', // Very Satisfied: Electric Violet/Purple
]

function SentimentChart({ data }) {
  const chartData = Object.entries(data || {}).map(
    ([name, value]) => ({
      name,
      value,
    })
  )

  const hasData = chartData.some((item) => item.value > 0)

  return (
    <div className="rounded-2xl border border-purple-100/90 bg-white p-6 shadow-xs transition hover:shadow-md">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-bold text-slate-900 tracking-tight">
            Sentiment Distribution
          </h3>
          <p className="mt-0.5 text-xs text-slate-500 font-medium">
            Customer feedback classified across 5 sentiment levels
          </p>
        </div>
      </div>

      <div className="mt-4 h-72">
        {!hasData ? (
          <div className="flex h-full items-center justify-center text-sm text-slate-400">
            No sentiment data to display yet.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="45%"
                innerRadius={68}
                outerRadius={104}
                paddingAngle={4}
                stroke="#ffffff"
                strokeWidth={2}
              >
                {chartData.map((entry, index) => (
                  <Cell
                    key={entry.name}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>

              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  border: '1px solid #e9d5ff',
                  borderRadius: '12px',
                  boxShadow: '0 10px 25px -5px rgba(139, 92, 246, 0.15)',
                  color: '#1e293b',
                  fontSize: '12px',
                  fontWeight: 600,
                  padding: '8px 12px',
                }}
                itemStyle={{ color: '#1e293b' }}
              />

              <Legend
                wrapperStyle={{
                  fontSize: '12px',
                  color: '#64748b',
                  fontWeight: 500,
                  paddingTop: '8px',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  )
}

export default SentimentChart
