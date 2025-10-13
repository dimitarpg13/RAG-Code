import axios from 'axios'

const BASE_API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'


/**
 * API client for repository indexing operations.
 */
export default class IndexingAPI {

    /**
     * Index a GitHub repository by URL.
     * 
     * @param githubUrl - The GitHub repository URL to index
     * @returns Promise resolving to the API response data
     * @throws Error if the API call fails
     */
    static async indexUrl(githubUrl: string) {
        const path = new URL('indexing/index', BASE_API).toString()
        const data = { github_url: githubUrl }

        try {
            const response = await axios.post(path, data)
            return response.data
        } catch (error) {
            throw new Error(`API call failed: ${error instanceof Error ? error.message : String(error)}`)
        }
    }

    /**
     * Retrieve all indexed repositories.
     * 
     * @returns Promise resolving to array of indexed repository data
     * @throws Error if the API call fails
     */
    static async getIndexedRepos() {
        const path = new URL('indexing/repos', BASE_API).toString()
        try {
            const response = await axios.get(path)
            return response.data
        } catch (error) {
            throw new Error(`API call failed: ${error instanceof Error ? error.message : String(error)}`
            )
        }
    }
}