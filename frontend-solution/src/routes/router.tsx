import { Navigate, useRoutes } from 'react-router'
import ChatBotPage from 'pages/ChatBotPage'
import IndexingPage from 'pages/IndexingPage'


/**
 * Main application router component.
 * 
 * Defines the routing structure for the application:
 * - /chatbot: Chat interface page
 * - /indexing: Repository indexing page
 * - * (all other routes): Redirects to /indexing
 * 
 * @returns JSX element containing the configured routes
 */
export default function Router() {
    return useRoutes([
        {
            path: '/chatbot',
            element: <ChatBotPage/>,
        },
        {
            path: '/indexing',
            element: <IndexingPage />,
        },
        { path: '*', element: <Navigate to="/indexing" replace /> }
    ])
}