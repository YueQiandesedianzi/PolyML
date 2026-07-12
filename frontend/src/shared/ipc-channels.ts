// Shared IPC channel names between main and renderer processes
export const IPC_CHANNELS = {
  SELECT_FILE: 'select-file',
  SELECT_DIRECTORY: 'select-directory',
  SELECT_SAVE_FILE: 'select-save-file',
  GET_APP_DATA_PATH: 'get-app-data-path',
  GET_PLATFORM: 'get-platform',
} as const
