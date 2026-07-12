import { app, BrowserWindow, dialog } from 'electron'
import { randomBytes } from 'crypto'
import { createServer } from 'net'
import { join } from 'path'
import { registerIpcHandlers } from './ipc-handlers'
import { PythonManager } from './python-manager'

let mainWindow: BrowserWindow | null = null
const pythonManager = new PythonManager()
const sessionToken = randomBytes(32).toString('hex')

async function reserveAvailableLoopbackPort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer()
    server.once('error', reject)
    server.listen(0, '127.0.0.1', () => {
      const address = server.address()
      if (!address || typeof address === 'string') {
        server.close(() => reject(new Error('Unable to allocate a loopback port')))
        return
      }
      const port = address.port
      server.close((error) => error ? reject(error) : resolve(port))
    })
  })
}

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 960,
    minHeight: 640,
    title: 'PolyML',
    webPreferences: {
      preload: join(__dirname, '../preload/preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    show: false,
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow?.show()
  })
  mainWindow.on('closed', () => {
    mainWindow = null
  })

  if (process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

app.whenReady().then(async () => {
  const backendPort = await reserveAvailableLoopbackPort()
  registerIpcHandlers({ port: backendPort, sessionToken })

  const backendDir = app.isPackaged
    ? join(process.resourcesPath, 'backend')
    : join(app.getAppPath(), '..', 'backend')
  try {
    await pythonManager.start(backendDir, backendPort, sessionToken)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    dialog.showErrorBox('PolyML backend startup failed', message)
  }

  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  pythonManager.stop()
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => {
  pythonManager.stop()
})
