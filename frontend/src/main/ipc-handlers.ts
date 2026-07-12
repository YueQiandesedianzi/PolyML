import { ipcMain, dialog, app, BrowserWindow } from 'electron'
import { join } from 'path'

export function registerIpcHandlers(): void {
  ipcMain.handle('select-file', async (_event, filters: Array<{name: string, extensions: string[]}>) => {
    const win = BrowserWindow.getFocusedWindow()
    if (!win) return null

    const result = await dialog.showOpenDialog(win, {
      properties: ['openFile'],
      filters: filters || [
        { name: 'Data Files', extensions: ['csv', 'xlsx', 'xls'] },
        { name: 'All Files', extensions: ['*'] },
      ],
    })

    return result.canceled ? null : result.filePaths[0]
  })

  ipcMain.handle('select-directory', async () => {
    const win = BrowserWindow.getFocusedWindow()
    if (!win) return null

    const result = await dialog.showOpenDialog(win, {
      properties: ['openDirectory'],
    })

    return result.canceled ? null : result.filePaths[0]
  })

  ipcMain.handle('select-save-file', async (_event, defaultName: string) => {
    const win = BrowserWindow.getFocusedWindow()
    if (!win) return null

    const result = await dialog.showSaveDialog(win, {
      defaultPath: defaultName,
      filters: [
        { name: 'Model Files', extensions: ['joblib'] },
        { name: 'JSON Files', extensions: ['json'] },
        { name: 'CSV Files', extensions: ['csv'] },
      ],
    })

    return result.canceled ? null : result.filePath
  })

  ipcMain.handle('get-app-data-path', () => {
    return join(app.getPath('userData'), 'PolyML')
  })

  ipcMain.handle('get-platform', () => {
    return process.platform
  })
}
