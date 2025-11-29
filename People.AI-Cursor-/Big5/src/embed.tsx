import React from 'react'
import ReactDOM from 'react-dom/client'
import { EmbedTest } from './components/EmbedTest'
import './index.css'
import './embed.css'

// Get theme from URL parameters
const urlParams = new URLSearchParams(window.location.search);
const theme = urlParams.get('theme') || 'light';

// Create container if it doesn't exist
let container = document.getElementById('big-five-embed');
if (!container) {
  container = document.createElement('div');
  container.id = 'big-five-embed';
  document.body.appendChild(container);
}

// Set theme
container.setAttribute('data-theme', theme);

ReactDOM.createRoot(container).render(
  <React.StrictMode>
    <EmbedTest />
  </React.StrictMode>,
)
