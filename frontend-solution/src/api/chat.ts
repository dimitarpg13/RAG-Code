import axios from 'axios'

const BASE_API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'


/**
 * Properties for sending a chat message.
 */
interface MessageProp {
    message: string
    namespace: string
    userName: string
}

/**
 * API client for chat operations.
 */
export default class ChatAPI {

    /**
     * Send a chat message to the backend.
     *
     * @param params - Message parameters
     * @param params.message - The chat message content
     * @param params.namespace - Pinecone namespace that scopes retrieval
     * @param params.userName - Username of the sender
     * @returns Promise resolving to the API response data
     * @throws Error if the API call fails
     */
    static async sendMessages({ message, namespace, userName }: MessageProp) {
        const path = new URL('chat/message', BASE_API).toString()
        const data = { message, namespace, username: userName }

        try {
            const response = await axios.post(path, data)
            return response.data
        } catch (error) {
            throw new Error(`API call failed: ${error instanceof Error ? error.message : String(error)}`
            )
        }
    }
}
