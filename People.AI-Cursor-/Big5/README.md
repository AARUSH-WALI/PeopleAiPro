# Big Five Personality Test

A modern, responsive implementation of the Big Five (OCEAN) personality test with AI-powered insights.

## Features

- 50-question personality assessment
- Real-time scoring and analysis
- AI-powered personality insights using Google's Gemini API
- Beautiful, responsive UI with Tailwind CSS
- Embeddable version for third-party websites

## Embedding the Test

To embed the personality test in your website, add the following code:

```html
<!-- Add the container for the test -->
<div id="big-five-embed"></div>

<!-- Add the script -->
<script type="module" src="https://your-domain.com/embed.js"></script>
```

### Customization Options

You can customize the embed by adding URL parameters:

- `theme`: Choose between 'light' and 'dark' (default: 'light')
- `lang`: Set the language (default: 'en')

Example:
```html
<script type="module" src="https://your-domain.com/embed.js?theme=dark&lang=en"></script>
```

## Development

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm run dev
```

3. Build for production:
```bash
npm run build
```

## Environment Variables

Create a `.env` file with:
```
VITE_GEMINI_API_KEY=your_gemini_api_key
```

## License

MIT License
