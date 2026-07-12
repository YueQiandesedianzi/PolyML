import { ChildProcess, spawn } from 'child_process'
import path from 'path'
import http from 'http'

export class PythonManager {
  private process: ChildProcess | null = null
  private port: number = 18921

  async start(backendDir: string, port: number = 18921): Promise<void> {
    this.port = port

    // Use conda run to execute within the polyml environment
    const args = [
      '-m', 'uvicorn', 'main:app',
      '--host', '127.0.0.1',
      '--port', String(port),
      '--log-level', 'info'
    ]

    // Detect conda env Python path
    const isWin = process.platform === 'win32'
    const homeDir = require('os').homedir()
    const condaEnvPython = isWin
      ? (process.env.POLYML_PYTHON || 'E:\\conda_envs\\polyml\\python.exe')
      : '/usr/local/envs/polyml/bin/python'

    console.log(`[PythonManager] Starting: ${condaEnvPython} ${args.join(' ')}`)
    console.log(`[PythonManager] Working directory: ${backendDir}`)

    this.process = spawn(condaEnvPython, args, {
      cwd: backendDir,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {
        ...process.env,
        OPENBLAS_NUM_THREADS: '1',
        MKL_NUM_THREADS: '1',
        NUMEXPR_NUM_THREADS: '1',
        OMP_NUM_THREADS: '1',
      },
    })

    this.process.stdout?.on('data', (data: Buffer) => {
      console.log(`[Python stdout] ${data.toString().trim()}`)
    })

    this.process.stderr?.on('data', (data: Buffer) => {
      console.log(`[Python stderr] ${data.toString().trim()}`)
    })

    this.process.on('exit', (code: number | null, signal: string | null) => {
      console.log(`[PythonManager] Process exited: code=${code}, signal=${signal}`)
      this.process = null
    })

    this.process.on('error', (err: Error) => {
      console.error(`[PythonManager] Process error: ${err.message}`)
      this.process = null
    })

    // Wait for backend to be healthy
    await this.waitForHealth(30000)
  }

  private waitForHealth(timeoutMs: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const startTime = Date.now()

      const check = () => {
        if (Date.now() - startTime > timeoutMs) {
          reject(new Error('Backend health check timed out'))
          return
        }

        const req = http.get(`http://127.0.0.1:${this.port}/api/health`, (res) => {
          if (res.statusCode === 200) {
            resolve()
          } else {
            setTimeout(check, 500)
          }
        })

        req.on('error', () => {
          setTimeout(check, 500)
        })

        req.setTimeout(1000, () => {
          req.destroy()
          setTimeout(check, 500)
        })
      }

      check()
    })
  }

  stop(): void {
    if (this.process) {
      console.log('[PythonManager] Stopping Python backend...')
      if (process.platform === 'win32') {
        spawn('taskkill', ['/pid', String(this.process.pid), '/f', '/t'])
      } else {
        this.process.kill('SIGTERM')
      }
      this.process = null
    }
  }

  getPort(): number {
    return this.port
  }

  isRunning(): boolean {
    return this.process !== null
  }
}
