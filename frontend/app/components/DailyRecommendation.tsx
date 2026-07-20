'use client';

import { useState, useEffect, useCallback } from 'react';

const API_BASE = 'http://localhost:8000';

type Dish = {
  name: string;
  category?: string;
  flavor?: string;
  cuisine?: string;
  ingredients?: string[];
  budget_tier?: string;
};

type Recommendation = {
  id: number;
  date: string;
  meal_type: string;
  dishes: Dish[];
  reasoning: string | null;
  generated_by: string;
  source_status: string;
  is_read: number;
};

type ReasoningData = {
  dishes?: { dish: string; reason: string }[];
  summary?: string;
};

function parseReasoning(raw: string | null): ReasoningData {
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    return { summary: raw };
  }
}

const MEAL_LABELS: Record<string, string> = {
  lunch: '午餐',
  dinner: '晚餐',
};

type Props = {
  userId: string;
  onClose: () => void;
};

export default function DailyRecommendation({ userId, onClose }: Props) {
  const [mealType, setMealType] = useState<'lunch' | 'dinner'>('lunch');
  const [data, setData] = useState<Recommendation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchRecommendation = useCallback(async (type: 'lunch' | 'dinner') => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `${API_BASE}/api/daily-recommendation?user_id=${encodeURIComponent(userId)}&meal_type=${type}`,
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      const json = await res.json();
      setData(json);
    } catch (e) {
      setError(e instanceof Error ? e.message : '加载失败');
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchRecommendation(mealType);
  }, [mealType, fetchRecommendation]);

  const handleRefresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/daily-recommendation/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, meal_type: mealType }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${res.status}`);
      }
      const json = await res.json();
      setData(json);
    } catch (e) {
      setError(e instanceof Error ? e.message : '生成失败');
    } finally {
      setLoading(false);
    }
  };

  const reasoning = parseReasoning(data?.reasoning ?? null);
  const dishReasons = Object.fromEntries(
    (reasoning.dishes ?? []).map(d => [d.dish, d.reason]),
  );

  const statusBadge: Record<string, string> = {
    success: 'text-green-600',
    partial: 'text-yellow-600',
    fallback: 'text-orange-500',
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center">
      <div className="bg-white rounded-xl shadow-xl w-[600px] max-h-[85vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b flex-shrink-0">
          <div className="flex items-center gap-3">
            <span className="font-semibold text-gray-700">今日推荐</span>
            <div className="flex gap-1">
              {(['lunch', 'dinner'] as const).map(t => (
                <button
                  key={t}
                  onClick={() => setMealType(t)}
                  className={[
                    'px-3 py-1 text-xs rounded-full transition-colors',
                    mealType === t
                      ? 'bg-orange-500 text-white'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200',
                  ].join(' ')}
                >
                  {MEAL_LABELS[t]}
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleRefresh}
              disabled={loading}
              className="text-xs text-gray-400 hover:text-gray-600 disabled:opacity-40 transition-colors"
              title="重新生成"
            >
              ↺ 换一换
            </button>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 text-xl leading-none"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4">
          {loading && (
            <div className="flex items-center justify-center h-32">
              <span className="text-gray-400 text-sm">加载中…</span>
            </div>
          )}

          {error && !loading && (
            <div className="flex items-center justify-center h-32">
              <span className="text-red-500 text-sm">{error}</span>
            </div>
          )}

          {data && !loading && (
            <div className="space-y-4">
              {/* Date & status */}
              <div className="flex items-center justify-between text-xs text-gray-400">
                <span>{data.date} · {MEAL_LABELS[data.meal_type] ?? data.meal_type}</span>
                <span className={statusBadge[data.source_status] ?? 'text-gray-400'}>
                  {data.source_status === 'success' ? '精准推荐'
                    : data.source_status === 'partial' ? '部分放宽'
                    : '兜底推荐'}
                </span>
              </div>

              {/* Dishes */}
              <div className="space-y-3">
                {(data.dishes ?? []).map((dish, i) => (
                  <div
                    key={i}
                    className="border border-gray-100 rounded-lg p-3 bg-gray-50 hover:bg-orange-50 transition-colors"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <span className="font-medium text-gray-800">{dish.name}</span>
                        {dish.category && (
                          <span className="ml-2 text-xs text-gray-400 bg-white border border-gray-200 rounded px-1.5 py-0.5">
                            {dish.category}
                          </span>
                        )}
                        {dish.flavor && (
                          <span className="ml-1 text-xs text-orange-400">{dish.flavor}</span>
                        )}
                      </div>
                      {dish.budget_tier && dish.budget_tier !== 'unknown' && (
                        <span className="text-xs text-gray-400 shrink-0">
                          {dish.budget_tier === 'low' ? '💰 经济' : dish.budget_tier === 'mid' ? '💰💰 适中' : '💰💰💰 高档'}
                        </span>
                      )}
                    </div>
                    {dishReasons[dish.name] && (
                      <p className="mt-1.5 text-xs text-gray-500">{dishReasons[dish.name]}</p>
                    )}
                    {dish.ingredients && dish.ingredients.length > 0 && (
                      <p className="mt-1 text-xs text-gray-400">
                        食材：{dish.ingredients.slice(0, 5).join('、')}
                        {dish.ingredients.length > 5 ? '…' : ''}
                      </p>
                    )}
                  </div>
                ))}
              </div>

              {/* Summary */}
              {reasoning.summary && (
                <div className="bg-orange-50 rounded-lg p-3 text-sm text-orange-700 leading-relaxed">
                  {reasoning.summary}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
