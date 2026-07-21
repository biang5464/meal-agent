'use client';

import { useState, useRef, useEffect, KeyboardEvent } from 'react';
import dynamic from 'next/dynamic';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiUrl } from './lib/api';

const PriceChart = dynamic(() => import('./components/PriceChart'), { ssr: false });
const DailyRecommendation = dynamic(() => import('./components/DailyRecommendation'), { ssr: false });

const USER_ID = 'test_user';

type Message = {
  role: 'user' | 'agent';
  content: string;
  error?: boolean;
};

/** 从 SSE block 中提取 data: 行的内容。 */
function extractPayload(block: string): string {
  return block
    .split(/\r?\n/)
    .filter(line => line.startsWith('data: '))
    .map(line => line.slice(6))
    .join('\n');
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPricePanel, setShowPricePanel] = useState(false);
  const [showDailyRec, setShowDailyRec] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  async function handleSend() {
    const text = input.trim();
    if (!text || loading) return;

    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setLoading(true);

    try {
      const res = await fetch(apiUrl('/recommend'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: USER_ID, message: text }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let replied = false;
      let finished = false;

      while (!finished) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // 按 SSE 事件边界切割，保留末尾不完整的块继续缓冲
        const blocks = buffer.split(/\r?\n\r?\n/);
        buffer = blocks.pop() ?? '';

        for (const block of blocks) {
          const payload = extractPayload(block);
          if (!payload) continue;

          if (payload === '[DONE]') { finished = true; break; }

          if (payload.startsWith('[ERROR]')) {
            setMessages(prev => [
              ...prev,
              { role: 'agent', content: payload.slice(7).trim() || '未知错误', error: true },
            ]);
            replied = true;
            finished = true;
            break;
          }

          // 首个 token 新建 agent 消息，后续 token 追加到同一条消息
          setMessages(prev => {
            const last = prev[prev.length - 1];
            if (replied && last && last.role === 'agent' && !last.error) {
              return [...prev.slice(0, -1), { ...last, content: last.content + payload }];
            }
            return [...prev, { role: 'agent', content: payload }];
          });
          replied = true;
        }
      }

      if (!replied) {
        setMessages(prev => [
          ...prev,
          { role: 'agent', content: '（未收到回复）', error: true },
        ]);
      }
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          role: 'agent',
          content: err instanceof Error ? err.message : '网络错误，请稍后重试',
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 shadow-sm flex-shrink-0 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-gray-800">饮食推荐助手</h1>
          <p className="text-xs text-gray-400 mt-0.5">user: {USER_ID}</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setShowDailyRec(true)}
            className="px-3 py-1.5 text-sm rounded bg-orange-500 text-white hover:bg-orange-600 transition-colors"
          >
            🍽️ 今日推荐
          </button>
          <button
            onClick={() => setShowPricePanel(true)}
            className="px-3 py-1.5 text-sm rounded bg-indigo-500 text-white hover:bg-indigo-600 transition-colors"
          >
            📈 价格走势
          </button>
        </div>
      </header>

      {/* Message list */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.length === 0 && !loading && (
          <p className="text-center text-gray-400 text-sm mt-24 select-none">
            发送消息开始对话
            <br />
            <span className="text-xs">
              例如："帮我推荐今天吃什么" / "西红柿多少钱" / "我不喜欢香菜"
            </span>
          </p>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={[
                'max-w-[75%] rounded-2xl px-4 py-3 text-sm leading-relaxed',
                msg.role === 'user'
                  ? 'bg-blue-500 text-white rounded-br-none whitespace-pre-wrap'
                  : msg.error
                  ? 'bg-red-50 text-red-600 border border-red-200 rounded-bl-none whitespace-pre-wrap'
                  : 'bg-white text-gray-800 shadow-sm border border-gray-100 rounded-bl-none',
              ].join(' ')}
            >
              {msg.role === 'user' || msg.error ? (
                msg.content
              ) : (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    table: ({node, ...props}) => (
                      <table className="border-collapse border border-gray-300 text-sm my-2" {...props} />
                    ),
                    th: ({node, ...props}) => (
                      <th className="border border-gray-300 px-3 py-1 bg-gray-100 font-semibold" {...props} />
                    ),
                    td: ({node, ...props}) => (
                      <td className="border border-gray-300 px-3 py-1" {...props} />
                    ),
                    strong: ({node, ...props}) => (
                      <strong className="font-semibold" {...props} />
                    ),
                    p: ({node, ...props}) => (
                      <p className="mb-2 last:mb-0" {...props} />
                    ),
                    h3: ({node, ...props}) => (
                      <h3 className="font-bold text-base mt-3 mb-1" {...props} />
                    ),
                    ul: ({node, ...props}) => (
                      <ul className="list-disc list-inside mb-2" {...props} />
                    ),
                    li: ({node, ...props}) => (
                      <li className="mb-0.5" {...props} />
                    ),
                  }}
                >
                  {msg.content}
                </ReactMarkdown>
              )}
            </div>
          </div>
        ))}

        {loading && messages[messages.length - 1]?.role !== 'agent' && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-100 rounded-2xl rounded-bl-none px-4 py-3 shadow-sm">
              <span className="text-gray-400 text-sm">思考中…</span>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="bg-white border-t border-gray-200 px-4 py-4 flex-shrink-0">
        <div className="flex gap-2 max-w-3xl mx-auto">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入消息，按 Enter 发送…"
            disabled={loading}
            className="flex-1 rounded-xl border border-gray-200 bg-white px-4 py-2.5 text-sm text-gray-900 outline-none
                       placeholder-gray-400
                       focus:border-blue-400 focus:ring-2 focus:ring-blue-50
                       disabled:opacity-50 disabled:cursor-not-allowed transition"
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="rounded-xl bg-blue-500 text-white px-5 py-2.5 text-sm font-medium
                       hover:bg-blue-600 active:bg-blue-700
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            发送
          </button>
        </div>
      </div>

      {/* 今日推荐弹出面板 */}
      {showDailyRec && (
        <DailyRecommendation
          userId={USER_ID}
          onClose={() => setShowDailyRec(false)}
        />
      )}

      {/* 价格走势弹出面板 */}
      {showPricePanel && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center">
          <div className="bg-white rounded-xl shadow-xl w-[800px] h-[500px] flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 border-b flex-shrink-0">
              <span className="font-semibold text-gray-700">价格走势</span>
              <button
                onClick={() => setShowPricePanel(false)}
                className="text-gray-400 hover:text-gray-600 text-xl leading-none"
              >
                ✕
              </button>
            </div>
            <div className="flex-1 min-h-0">
              <PriceChart />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
