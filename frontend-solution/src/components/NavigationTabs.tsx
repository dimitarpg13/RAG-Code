import React from 'react';
import { Tabs, Tab, } from '@mui/material'
import { useNavigate, useLocation } from 'react-router'


/**
 * Navigation tabs component for switching between main application pages.
 * 
 * Provides a Material-UI tab interface that allows users to navigate between:
 * - Indexing page (tab index 0)
 * - Chatbot page (tab index 1)
 * 
 * The active tab is determined by the current route path.
 * 
 * @returns JSX element containing the navigation tabs
 */
export default function NavigationTabs() {
    // Hook for programmatic navigation between routes
    const navigate = useNavigate()
    // Hook to get current location/path information
    const location = useLocation()
    // Determine active tab based on current path (0 = indexing, 1 = chatbot)
    const value = location.pathname === '/indexing' ? 0 : 1

    /**
     * Handle tab change events and navigate to the corresponding route.
     * 
     * @param _ - React synthetic event (unused)
     * @param newValue - Index of the selected tab (0 for indexing, 1 for chatbot)
     */
    const handleChange = (
        _: React.SyntheticEvent, newValue: number) => {
        // Navigate to indexing page when first tab (index 0) is selected
        if (newValue === 0) {
            navigate('/indexing')
        } else {
            // Navigate to chatbot page for any other tab selection
            navigate('/chatbot')
        }
    }

    return <Tabs
        value={value}  // Current active tab index
        onChange={handleChange}  // Handler for tab selection changes
        centered  // Center the tabs horizontally
        sx={{ width: '100%' }}>  // Full width styling
        <Tab label="Indexing" />  {/* First tab - Repository indexing */}
        <Tab label="Chatbot" />   {/* Second tab - Chat interface */}
    </Tabs>
}