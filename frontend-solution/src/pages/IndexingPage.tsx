
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
        if (!url.trim()) return
        
        setLoading(true)
        try {
            const response = await IndexingAPI.indexUrl(url)
            console.log('Indexing started:', response)
        } catch (error) {
            console.error('Indexing error:', error)
        } finally {
            setLoading(false)
        }
    }

    return (
        <Box sx={{ padding: 4, maxWidth: 600, margin: '0 auto' }}>
            <Typography variant="h6" gutterBottom>
                Github Repo URL
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                <TextField
                    label="Github Repo URL"
                    variant="outlined"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="https://github.com/huggingface/transformers"
                    fullWidth
                />
                <Button
                    variant="contained"
                    onClick={handleCrawl}
                    disabled={loading || !url.trim()}
                >
                    {loading ? 'Indexing...' : 'Index'}
                </Button>
            </Box>
        </Box>
    )
}
