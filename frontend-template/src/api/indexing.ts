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
            /* TODO: implement indexUrl. Use axios.post function to send the GitHub repo URL to the indexing/index endpoint.
             await axios.post(path, data)*/
            const response = null
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
            /* Use the axios.get function to send a request to indexing/repos. 
            await axios.get(path) */ 
            const response = null
            return response.data
        } catch (error) {
            throw new Error(`API call failed: ${error instanceof Error ? error.message : String(error)}`
            )
        }
    }
}