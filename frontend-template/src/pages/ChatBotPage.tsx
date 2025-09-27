// ChatBotPage.tsx
import ChatBot, { type Params } from "react-chatbotify";
import { Box, FormControl, InputLabel, Select, Typography, MenuItem } from '@mui/material';
import { useState, useEffect } from 'react';
import IndexingAPI from 'api/indexing';
import ChatAPI from 'api/chat';

/**
 * Interface representing an indexed GitHub repository.
 */
interface IndexedRepo {
    github_url: string;
    namespace: string;
    indexed_at: string;
}

/**
 * Chatbot page component for interacting with indexed GitHub repositories.
 * 
 * Features:
 * - Dropdown to select from previously indexed repositories
 * - Interactive chatbot that can answer questions about the selected repository
 * - Uses RAG (Retrieval-Augmented Generation) to provide contextual responses
 * 
 * The component loads all indexed repositories on mount and allows users to
 * chat about the selected repository's codebase.
 * 
 * @returns JSX element containing the repository selector and chatbot interface
 */
export default function ChatBotPage() {
    const [repos, setRepos] = useState<IndexedRepo[]>([]);
    const [selectedNamespace, setSelectedNamespace] = useState('');


    useEffect(() => {
        /* TODO: load the repos */
    }, []);

    const settings = {
        general: {
            embedded: true, showFooter: false, showHeader: false
        },
    }

    /**
     * Handle user input from the chatbot and send it to the backend API.
     * 
     * @param params - Parameters from the chatbot containing user input
     * @returns Promise resolving to the bot's response message
     */
    const handleUserInput = async (params: Params) => {
        /* TODO: Let’s implement the function that handles a new user’s input by sending the input to the chat API.
        
        In the handleUserInput function, the chatbot will not process the input if a GitHub URL is not selected. 
        If it is selected, it will send the user’s message to the ChatAPI.sendMessages API. 
        For simplicity, I am calling the current userID “test_id”.
        */ 
    }

    const flow = {
        /* TODO: The flow is defined as such:
        - The user is presented with an introductory message: "Hello! I can help you with questions about a Github Repo. 
        Please select a repository from the dropdown above to get started."
        - Then the specified path is “chat_loop“, which means we continue the flow with the chat_loop
        - The message is handled with handleUserInput
        */

    }

    return (
        <Box>
            <Box>
                <Typography>
                </Typography>
                <FormControl >
                    <InputLabel></InputLabel>
                    <Select>
                    </Select>
                </FormControl>
            </Box>
            <Box >
                <ChatBot/>
            </Box>
        </Box>
    )
}