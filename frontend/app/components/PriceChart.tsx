'use client'

import { useEffect, useState } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import { apiUrl } from '../lib/api'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

const PLATFORM_COLORS: Record<string, string> = {
  woolworths: '#007B40',
  umall: '#E4002B',
  coles: '#CC0000',
  jbhifi: '#FF6200',
}

const PLATFORM_DISPLAY_NAMES: Record<string, string> = {
  woolworths: 'Woolworths',
  umall: 'Umall',
  coles: 'Coles',
  jbhifi: 'JB Hi-Fi',
}

interface PricePoint {
  date: string
  price: number
  unit?: string
}

interface ChartData {
  query_term: string
  mode: string
  comparable?: boolean
  unit?: string
  data: PricePoint[] | Record<string, PricePoint[]>
}

export default function PriceChart() {
  const [terms, setTerms] = useState<string[]>([])
  const [selectedTerm, setSelectedTerm] = useState<string>('')
  const [mode, setMode] = useState<'single' | 'multi'>('single')
  const [chartData, setChartData] = useState<ChartData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string>('')

  useEffect(() => {
    fetch(apiUrl('/api/tracked-terms'))
      .then(r => r.json())
      .then(d => {
        setTerms(d.terms || [])
        if (d.terms?.length > 0) setSelectedTerm(d.terms[0])
      })
      .catch(() => setError('无法加载商品列表'))
  }, [])

  useEffect(() => {
    if (!selectedTerm) return
    setLoading(true)
    setError('')
    fetch(apiUrl(`/api/price-history/${encodeURIComponent(selectedTerm)}?mode=${mode}&days=30`))
      .then(r => r.json())
      .then(d => setChartData(d))
      .catch(() => setError('加载价格数据失败'))
      .finally(() => setLoading(false))
  }, [selectedTerm, mode])

  const buildChartDataset = () => {
    if (!chartData) return null

    if (mode === 'single' && chartData.comparable && Array.isArray(chartData.data)) {
      const points = chartData.data as PricePoint[]
      return {
        labels: points.map(p => p.date),
        datasets: [{
          label: `${selectedTerm} 最低价 (${chartData.unit})`,
          data: points.map(p => p.price),
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99,102,241,0.1)',
          tension: 0.3,
        }]
      }
    }

    const byPlatform = chartData.data as Record<string, PricePoint[]>
    const allDates = Array.from(
      new Set(Object.values(byPlatform).flat().map(p => p.date))
    ).sort()

    const datasets = Object.entries(byPlatform).map(([platform, points]) => {
      const safePoints = Array.isArray(points) ? points : []
      const priceMap = Object.fromEntries(safePoints.map(p => [p.date, p.price]))
      const unit = safePoints[0]?.unit || ''
      return {
        label: `${PLATFORM_DISPLAY_NAMES[platform] || platform} (${unit})`,
        data: allDates.map(d => priceMap[d] ?? null),
        borderColor: PLATFORM_COLORS[platform] || '#888',
        backgroundColor: 'transparent',
        tension: 0.3,
        spanGaps: true,
      }
    })

    return { labels: allDates, datasets }
  }

  const dataset = buildChartDataset()

  return (
    <div className="flex flex-col gap-4 p-4 h-full">
      {/* 控制栏 */}
      <div className="flex items-center gap-3 flex-wrap">
        <select
          value={selectedTerm}
          onChange={e => setSelectedTerm(e.target.value)}
          className="border rounded px-3 py-1.5 text-sm bg-white"
        >
          {terms.map(t => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>

        <div className="flex rounded border overflow-hidden text-sm">
          <button
            onClick={() => setMode('single')}
            className={`px-3 py-1.5 ${mode === 'single' ? 'bg-indigo-500 text-white' : 'bg-white text-gray-600'}`}
          >
            单线
          </button>
          <button
            onClick={() => setMode('multi')}
            className={`px-3 py-1.5 ${mode === 'multi' ? 'bg-indigo-500 text-white' : 'bg-white text-gray-600'}`}
          >
            多线
          </button>
        </div>

        <span className="text-xs text-gray-400">最近30天</span>
      </div>

      {chartData && chartData.comparable === false && mode === 'single' && (
        <div className="text-xs text-amber-600 bg-amber-50 rounded px-3 py-2">
          各平台单位不同，无法直接比价，已按平台分别显示
        </div>
      )}

      <div className="flex-1 min-h-0">
        {loading && (
          <div className="flex items-center justify-center h-full text-gray-400 text-sm">
            加载中...
          </div>
        )}
        {error && (
          <div className="flex items-center justify-center h-full text-red-400 text-sm">
            {error}
          </div>
        )}
        {!loading && !error && dataset && (
          <Line
            data={dataset}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: { position: 'top' },
                title: {
                  display: true,
                  text: `${selectedTerm} 价格走势`,
                },
                tooltip: {
                  callbacks: {
                    label: ctx => `${ctx.dataset.label}: $${ctx.parsed.y?.toFixed(2) ?? 'N/A'}`,
                  }
                }
              },
              scales: {
                y: {
                  beginAtZero: false,
                  ticks: {
                    callback: val => `$${val}`,
                  }
                }
              }
            }}
          />
        )}
        {!loading && !error && !dataset && (
          <div className="flex items-center justify-center h-full text-gray-400 text-sm">
            暂无价格数据
          </div>
        )}
      </div>
    </div>
  )
}
