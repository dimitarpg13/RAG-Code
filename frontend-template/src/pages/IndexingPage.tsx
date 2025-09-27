
import { useState } from 'react'
import { TextField, Button, Box, Typography } from '@mui/material'
import IndexingAPI from 'api/indexing'

/**
 * Repository indexing page component.
 * 
 * Provides a user interface for indexing GitHub repositories:
 * - Input field for GitHub repository URL
 * - Index button to trigger the indexing process
 * - Loading state management during indexing
 * 
 * @returns JSX element containing the indexing interface
 */
export default function IndexingPage() {
    const [url, setUrl] = useState('')
    const [loading, setLoading] = useState(false)

    /**
     * Handle the repository indexing process.
     * 
     * Validates the URL input, calls the indexing API, and manages
     * the loading state during the operation.
     */
    const handleCrawl = async () => {
        /* TODO: When the url is ready, the user can click on the button to use the IndexingAPI.indexUrl API. Let’s implement the function that takes care of this logic:
        - If the url is empty, nothing happens
        - If it is not, we set loading to true, and call the IndexingAPI.indexUrl API
        - Once we got the answer, error or not, we set the loading back to false. */
    }

    return (
        <Box>
            <Typography>
            </Typography>
            <Box>
                <TextField/>
                <Button></Button>
            </Box>
        </Box>
    )
}
