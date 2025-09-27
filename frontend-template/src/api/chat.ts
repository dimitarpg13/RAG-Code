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
    static async sendMessages({  message, namespace, userName }: MessageProp) {
        /* TODO: Create the data dictionary consistent with the way the attributes are 
        named in backend/app/chat/schemas.py, and send the data with a post request to chat/message. */
        const path = null
        const data = null
    }
}