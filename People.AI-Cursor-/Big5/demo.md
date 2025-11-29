# Embedding the Big Five Personality Test

This guide explains how to embed the Big Five personality test into any website.

## Quick Start

Add the following code snippet to your HTML where you want the test to appear:

```html
<div id="big-five-embed"></div>
<script type="module" src="https://your-deployment-url/embed-test.js"></script>
```

That's it! The test will automatically render in the container.

## Customization

### Theme

You can customize the theme by adding a `theme` parameter to the script URL:

```html
<div id="big-five-embed"></div>
<script type="module" src="https://your-deployment-url/embed-test.js?theme=dark"></script>
```

Available themes:
- `light` (default)
- `dark`

### Container

The test will automatically create a container if one doesn't exist, but for better control over placement, we recommend adding your own container:

```html
<div id="big-five-embed" style="width: 100%; max-width: 800px; margin: 0 auto;"></div>
```

## Example Implementation

Here's a complete example of how to embed the test in your webpage:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Website - Personality Test</title>
</head>
<body>
    <h1>Take the Big Five Personality Test</h1>
    
    <!-- Personality Test Container -->
    <div id="big-five-embed"></div>
    <script type="module" src="https://your-deployment-url/embed-test.js"></script>
</body>
</html>
```

## Notes

- The embed script is lightweight and will only load the necessary resources when the test is in view
- The test is responsive and will adapt to its container's width
- Results can be displayed within the embedded test interface
- The test maintains its own state and won't interfere with your website's JavaScript

## Support

For issues or questions about embedding the test, please visit our [GitHub repository](https://github.com/yourusername/big5) or create an issue.
