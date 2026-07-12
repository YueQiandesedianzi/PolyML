import React, { Component, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex flex-col items-center justify-center h-full p-8 text-center">
          <div className="text-4xl mb-4">⚠️</div>
          <h2 className="text-lg font-semibold text-gray-800 mb-2">页面出错了</h2>
          <p className="text-sm text-gray-500 mb-4 max-w-md">{this.state.error.message}</p>
          <button
            onClick={() => this.setState({ error: null })}
            className="bg-primary-600 text-white px-4 py-2 rounded-md text-sm hover:bg-primary-700"
          >
            重试
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
