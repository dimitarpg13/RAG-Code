// Core routing components
import Router from 'routes/router'
import { BrowserRouter } from "react-router"
// Material-UI layout component
import { Box } from '@mui/material'
// Custom navigation component
import NavigationTabs from 'components/NavigationTabs'

/**
 * Main application component.
 * 
 * Sets up the routing context and renders the main layout structure:
 * - BrowserRouter: Enables client-side routing
 * - NavigationTabs: Tab-based navigation between pages
 * - Router: Route definitions and page rendering
 * 
 * @returns JSX element containing the full application structure
 */
function App() {
  return <BrowserRouter>  {/* Enable browser-based routing */}
    <Box sx={{ width: '100vw', minHeight: '100vh' }}>  {/* Full viewport container */}
      <NavigationTabs />  {/* Navigation tabs at the top */}
      <Router />          {/* Main content area with route-based rendering */}
    </Box>
  </BrowserRouter>
}

export default App
