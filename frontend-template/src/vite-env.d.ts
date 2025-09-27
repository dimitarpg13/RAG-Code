// Imports Vite's base environment types (like import.meta.env.MODE, import.meta.env.DEV, etc.)
/// <reference types="vite/client" />

// Custom Environment Variables
// Tells TypeScript "Hey, there's also a VITE_API_BASE_URL variable of type string"
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string  // ← Your custom variable
}

// Connecting the Pieces
interface ImportMeta {
  readonly env: ImportMetaEnv // ← Links your custom types to import.meta.env
}