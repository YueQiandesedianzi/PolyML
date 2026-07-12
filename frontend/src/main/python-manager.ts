import { ChildProcess, spawn, spawnSync } from 'child_process'
import fs from 'fs'
import http from 'http'
import os from 'os'
import path from 'path'

export class PythonManager {
  private process: ChildProcess | null = null
  private port = 0
  private sessionToken = ''

  async start(backendDir: string, port: number, sessionToken: string): Promise<void> {
    this.port = port
    this.sessionToken = sessionToken
    const python = this.discoverPython(backendDir)
    const args = ['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', String(port), '--log-level', 'info']

    console.log(`[PythonManager] Starting: ${python} ${args.join(' ')}`)
    console.log(`[PythonManager] Working directory: ${backendDir}`)
    this.process = spawn(python, args, {
      cwd: backendDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        POLYML_SESSION_TOKEN: sessionToken,
        PYTHONUTF8: '1',
        PYTHONIOENCODING: 'utf-8',
        OPENBLAS_NUM_THREADS: '1',
        MKL_NUM_THREADS: '1',
        NUMEXPR_NUM_THREADS: '1',
        OMP_NUM_THREADS: '1',
      },
    })

    this.process.stdout?.on('data', (data: Buffer) => console.log(`[Python stdout] ${data.toString().trim()}`))
    this.process.stderr?.on('data', (data: Buffer) => console.log(`[Python stderr] ${data.toString().trim()}`))
    this.process.on('exit', (code, signal) => {
      console.log(`[PythonManager] Process exited: code=${code}, signal=${signal}`)
      this.process = null
    })
    this.process.on('error', (error) => {
      console.error(`[PythonManager] Process error: ${error.message}`)
      this.process = null
    })

    try {
      await this.waitForHealth(30_000)
    } catch (error) {
      this.stop()
      throw error
    }
  }

  private discoverPython(backendDir: string): string {
    const isWin = process.platform === 'win32'
    const executable = isWin ? 'python.exe' : 'bin/python'
    const candidates: string[] = []
    if (process.env.POLYML_PYTHON) candidates.push(process.env.POLYML_PYTHON)
    if (process.env.CONDA_PREFIX) candidates.push(path.join(process.env.CONDA_PREFIX, executable))

    const condaCommand = process.env.CONDA_EXE || 'conda'
    const conda = spawnSync(condaCommand, ['env', 'list', '--json'], { encoding: 'utf8', timeout: 10_000 })
    if (conda.status === 0 && conda.stdout) {
      try {
        const envs = (JSON.parse(conda.stdout).envs || []) as string[]
        const polyml = envs.find((envPath) => path.basename(envPath).toLowerCase() === 'polyml')
        if (polyml) candidates.push(path.join(polyml, executable))
      } catch { /* continue with common locations */ }
    }

    const home = os.homedir()
    for (const root of ['miniconda3', 'anaconda3', 'miniforge3', 'mambaforge']) {
      candidates.push(path.join(home, root, 'envs', 'polyml', executable))
    }
    candidates.push(isWin ? 'python.exe' : 'python3')

    for (const candidate of [...new Set(candidates)]) {
      if ((candidate.includes(path.sep) && !fs.existsSync(candidate)) || !this.verifyPython(candidate)) continue
      return candidate
    }
    const environmentFile = path.join(backendDir, 'environment.yml')
    throw new Error(
      'PolyML Python environment was not found or does not meet the required dependencies.\n' +
      `Create it with:\nconda env create -f "${environmentFile}"\n` +
      "Then restart PolyML, or set POLYML_PYTHON to the environment's Python executable."
    )
  }

  private verifyPython(candidate: string): boolean {
    const check = spawnSync(candidate, [
      '-c',
      'import sys, fastapi, rdkit, xgboost, optuna, shap, pyDOE3; assert sys.version_info[:2] == (3, 11)',
    ], { encoding: 'utf8', timeout: 15_000 })
    if (check.status !== 0) {
      console.warn(`[PythonManager] Rejected ${candidate}: ${(check.stderr || check.stdout || '').trim()}`)
      return false
    }
    return true
  }

  private waitForHealth(timeoutMs: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const started = Date.now()
      const check = () => {
        if (Date.now() - started > timeoutMs) {
          reject(new Error(`Backend health check timed out on port ${this.port}. Check Python dependencies and local firewall rules.`))
          return
        }
        const request = http.get({
          hostname: '127.0.0.1',
          port: this.port,
          path: '/api/health',
          headers: { 'X-PolyML-Token': this.sessionToken },
          timeout: 1_000,
        }, (response) => {
          let body = ''
          response.on('data', (chunk) => { body += chunk })
          response.on('end', () => {
            try {
              const health = JSON.parse(body)
              if (response.statusCode === 200 && health.status === 'ok' && health.version) resolve()
              else setTimeout(check, 500)
            } catch { setTimeout(check, 500) }
          })
        })
        request.on('error', () => setTimeout(check, 500))
        request.on('timeout', () => request.destroy())
      }
      check()
    })
  }

  stop(): void {
    if (!this.process) return
    console.log('[PythonManager] Stopping Python backend...')
    if (process.platform === 'win32') spawn('taskkill', ['/pid', String(this.process.pid), '/f', '/t'])
    else this.process.kill('SIGTERM')
    this.process = null
  }

  getPort(): number { return this.port }
  isRunning(): boolean { return this.process !== null }
}
