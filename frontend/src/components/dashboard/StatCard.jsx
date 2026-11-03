function StatCard({ title, value, subtitle, icon: Icon, color }) {
  return (
    <div className="rounded-2xl border border-purple-100/90 bg-white p-5 shadow-xs transition hover:border-purple-200 hover:shadow-md">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-slate-500">
          {title}
        </p>

        <div className={`rounded-xl p-2.5 shadow-xs ${color}`}>
          <Icon size={20} />
        </div>
      </div>

      <h2 className="mt-4 text-3xl font-extrabold text-slate-900 tracking-tight">
        {value}
      </h2>

      <p className="mt-1 text-xs text-slate-400 font-medium">
        {subtitle}
      </p>
    </div>
  )
}

export default StatCard
