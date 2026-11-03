import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from 'recharts'

const BAR_COLORS = [
  '#8b5cf6', // Violet for Text Only
  '#d946ef', // Fuchsia for Image Only
  '#f59e0b', // Amber for Multimodal
]

function ModelUsageChart({ data }) {
  const chartData = [
    { name: 'Text Only', count: data?.text_only || 0 },
    { name: 'Image Only', count: data?.image_only || 0 },
    { name: 'Multimodal', count: data?.multimodal || 0 },
  ]

  return (
    <div className="rounded-2xl border border-purple-100/90 bg-white p-6 shadow-xs transition hover:shadow-md">
      <div>
        <h3 className="text-lg font-bold text-slate-900 tracking-tight">
          Model Usage Distribution
        </h3>
        <p className="mt-0.5 text-xs text-slate-500 font-medium">
          Predictions breakdown by model pipeline architecture
        </p>
      </div>

      <div className="mt-4 h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#f1f5f9"
              vertical={false}
            />

            <XAxis
              dataKey="name"
              stroke="#94a3b8"
              fontSize={12}
              fontWeight={500}
              tickLine={false}
              axisLine={{ stroke: '#e2e8f0' }}
            />

            <YAxis
              stroke="#94a3b8"
              fontSize={12}
              fontWeight={500}
              allowDecimals={false}
              tickLine={false}
              axisLine={{ stroke: '#e2e8f0' }}
            />

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
              cursor={{ fill: 'rgba(243, 232, 255, 0.3)' }}
            />

            <Bar
              dataKey="count"
              radius={[8, 8, 0, 0]}
              maxBarSize={48}
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={BAR_COLORS[index % BAR_COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default ModelUsageChart
