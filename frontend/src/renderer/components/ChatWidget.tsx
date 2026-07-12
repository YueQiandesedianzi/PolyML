import React, { useState, useRef, useEffect } from 'react'
import { sendChatMessage, type ChatMessage } from '@/renderer/services/api'
import { useProjectStore } from '@/renderer/store/projectStore'

export const ChatWidget: React.FC = () => {
  const { currentProject } = useProjectStore()
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    if (open) inputRef.current?.focus()
  }, [open])

  const handleSend = async () => {
    const text = input.trim()
    if (!text || loading) return

    const userMsg: ChatMessage = { role: 'user', content: text }
    const updated = [...messages, userMsg]
    setMessages(updated)
    setInput('')
    setError(null)
    setLoading(true)

    try {
      const reply = await sendChatMessage(updated, currentProject?.id)
      setMessages([...updated, { role: 'assistant', content: reply }])
    } catch (e: any) {
      const errMsg = e.response?.data?.detail || e.message || '请求失败'
      setError(errMsg)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <>
      {/* Chat panel */}
      {open && (
        <div className="fixed bottom-20 right-6 w-[380px] max-h-[520px] bg-white rounded-xl shadow-2xl border border-gray-200 flex flex-col overflow-hidden z-50">
          {/* Header */}
          <div className="bg-primary-600 text-white px-4 py-3 flex items-center justify-between">
            <div>
              <div className="text-sm font-semibold">PolyML 智能助手</div>
              {currentProject && (
                <div className="text-[10px] text-primary-200 mt-0.5">
                  项目: {currentProject.name}
                </div>
              )}
            </div>
            <button
              onClick={() => setOpen(false)}
              className="text-primary-200 hover:text-white text-lg leading-none"
            >
              ✕
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-gray-50">
            {messages.length === 0 && (
              <div className="text-center text-xs text-gray-400 mt-8">
                <div className="text-2xl mb-2">💬</div>
                <p>你好！我是 PolyML 智能助手。</p>
                <p className="mt-1">可以问我任何聚合物科学或平台使用的问题。</p>
              </div>
            )}
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[85%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap ${
                    msg.role === 'user'
                      ? 'bg-primary-500 text-white'
                      : 'bg-white text-gray-800 border border-gray-200'
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-white text-gray-500 border border-gray-200 rounded-lg px-3 py-2 text-sm animate-pulse">
                  思考中...
                </div>
              </div>
            )}
            {error && (
              <div className="text-center text-xs text-red-500 bg-red-50 rounded-lg px-3 py-2">
                {error}
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-gray-200 p-3 bg-white">
            <div className="flex gap-2 items-end">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="输入问题... (Enter 发送)"
                rows={1}
                className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-primary-500 max-h-20"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || loading}
                className="bg-primary-600 text-white rounded-lg px-3 py-2 text-sm font-medium hover:bg-primary-700 disabled:opacity-40 transition shrink-0"
              >
                发送
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Floating button */}
      <button
        onClick={() => setOpen(!open)}
        className="fixed bottom-6 right-6 w-14 h-14 bg-primary-500 hover:bg-primary-600 text-white rounded-full shadow-lg flex items-center justify-center z-50 transition-transform hover:scale-105"
        aria-label="智能助手"
      >
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
      </button>
    </>
  )
}
