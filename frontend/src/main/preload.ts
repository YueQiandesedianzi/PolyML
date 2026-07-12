import { contextBridge, ipcRenderer } from 'electron'

export interface FileFilter {
  name: string
  extensions: string[]
}

export interface ElectronAPI {
  selectFile: (filters?: FileFilter[]) => Promise<string | null>
  selectDirectory: () => Promise<string | null>
  selectSaveFile: (defaultName: string) => Promise<string | null>
  getAppDataPath: () => Promise<string>
  getPlatform: () => Promise<string>
}

const electronAPI: ElectronAPI = {
  selectFile: (filters) => ipcRenderer.invoke('select-file', filters),
  selectDirectory: () => ipcRenderer.invoke('select-directory'),
  selectSaveFile: (defaultName) => ipcRenderer.invoke('select-save-file', defaultName),
  getAppDataPath: () => ipcRenderer.invoke('get-app-data-path'),
  getPlatform: () => ipcRenderer.invoke('get-platform'),
}

contextBridge.exposeInMainWorld('electronAPI', electronAPI)
