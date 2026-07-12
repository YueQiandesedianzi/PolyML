interface Window {
  electronAPI?: {
    getBackendConfig: () => Promise<{ baseUrl: string; sessionToken: string }>
  }
}
