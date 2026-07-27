'use client';

import { useState, useEffect, useCallback } from 'react';
import { apiUrl } from '../lib/api';

type Dish = {
  name: string;
  category?: string;
  flavor?: string;
  cuisine?: string;
  ingredients?: string[];
  budget_tier?: string;
  steps?: string[] | string;
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

function renderSteps(steps?: string[] | string) {
  if (!steps) {
    return <p className="text-xs text-gray-400">详细做法暂未收录</p>;
  }
  if (Array.isArray(steps)) {
    if (steps.length === 0) {
      return <p className="text-xs text-gray-400">详细做法暂未收录</p>;
    }
    return (
      <ol className="text-xs text-gray-600 space-y-1 list-decimal list-inside">
        {steps.map((step, i) => (
          <li key={i} className="leading-relaxed break-words">{step}</li>
        ))}
      </ol>
    );
  }
  // string fallback (historical data)
  if (!steps.trim()) {
    return <p className="text-xs text-gray-400">详细做法暂未收录</p>;
  }
  return <p className="text-xs text-gray-600 leading-relaxed break-words">{steps}</p>;
}

const MEAL_LABELS: Record<string, string> = {
  lunch: '午餐',
  dinner: '晚餐',
};

type Props = {
  onClose: () => void;
};

export default function DailyRecommendation({ onClose }: Props) {
  const [mealType, setMealType] = useState<'lunch' | 'dinner'>('lunch');
  const [data, setData] = useState<Recommendation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedDish, setExpandedDish] = useState<string | null>(null);

  // Reset expanded state immediately when switching meal type
  useEffect(() => {
    setExpandedDish(null);
  }, [mealType]);

  // Reset expanded state whenever recommendation data is replaced
  useEffect(() => {
    setExpandedDish(null);
  }, [data]);

  const fetchRecommendation = useCallback(async (type: 'lunch' | 'dinner') => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        apiUrl(`/api/daily-recommendation?meal_type=${type}`),
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
  }, []);

  useEffect(() => {
    fetchRecommendation(mealType);
  }, [mealType, fetchRecommendation]);

  const handleRefresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(apiUrl('/api/daily-recommendation/generate'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ meal_type: mealType }),
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
                  type="button"
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
              type="button"
              onClick={handleRefresh}
              disabled={loading}
              className="text-xs text-gray-400 hover:text-gray-600 disabled:opacity-40 transition-colors"
              title="重新生成"
            >
              ↺ 换一换
            </button>
            <button
              type="button"
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
                {(data.dishes ?? []).map((dish, i) => {
                  const dishKey = dish.name || String(i);
                  const isExpanded = expandedDish === dishKey;
                  const panelId = `dish-panel-${i}`;

                  return (
                    <div
                      key={dishKey}
                      className="border border-gray-100 rounded-lg bg-gray-50 overflow-hidden"
                    >
                      {/* Clickable card header */}
                      <button
                        type="button"
                        onClick={() => setExpandedDish(isExpanded ? null : dishKey)}
                        aria-expanded={isExpanded}
                        aria-controls={panelId}
                        className="w-full text-left p-3 hover:bg-orange-50 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-orange-400 focus-visible:ring-inset"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex-1 min-w-0">
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
                          <div className="flex items-center gap-2 shrink-0">
                            {dish.budget_tier && dish.budget_tier !== 'unknown' && (
                              <span className="text-xs text-gray-400">
                                {dish.budget_tier === 'low' ? '💰 经济' : dish.budget_tier === 'mid' ? '💰💰 适中' : '💰💰💰 高档'}
                              </span>
                            )}
                            <span className="text-xs text-gray-400 select-none">
                              {isExpanded ? '收起 ↑' : '做法 ↓'}
                            </span>
                          </div>
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
                      </button>

                      {/* Expandable detail panel */}
                      {isExpanded && (
                        <div
                          id={panelId}
                          className="px-3 pb-3 pt-2 border-t border-gray-100 space-y-3"
                        >
                          {/* Full ingredients */}
                          <div>
                            <p className="text-xs font-medium text-gray-600 mb-1">食材</p>
                            {dish.ingredients && dish.ingredients.length > 0 ? (
                              <ul className="text-xs text-gray-500 space-y-0.5 list-disc list-inside">
                                {dish.ingredients.map((ing, j) => (
                                  <li key={j}>{ing}</li>
                                ))}
                              </ul>
                            ) : (
                              <p className="text-xs text-gray-400">食材信息暂未收录</p>
                            )}
                          </div>

                          {/* Cooking steps */}
                          <div>
                            <p className="text-xs font-medium text-gray-600 mb-1">做法</p>
                            {renderSteps(dish.steps)}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
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
