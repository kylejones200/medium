# Technical Portfolio

This repository contains a collection of technical articles on energy systems, mining operations, and infrastructure analytics.

## GitHub Pages Setup

This repository is configured to automatically deploy the `poster` folder to GitHub Pages.

### Setup Instructions

1. **Push to GitHub**: Push this repository to GitHub
   ```bash
   git add .
   git commit -m "Setup GitHub Pages"
   git push origin main
   ```

2. **Enable GitHub Pages**:
   - Go to your repository on GitHub
   - Navigate to **Settings** → **Pages**
   - Under "Build and deployment":
     - Source: Select **GitHub Actions**
   - The site will automatically deploy when you push to the `main` branch

3. **Access Your Site**:
   - Your site will be available at: `https://<username>.github.io/<repository-name>/`
   - The deployment typically takes 1-2 minutes

### What Gets Deployed

Only the contents of the `poster` folder are deployed to GitHub Pages, including:
- `index.html` - Landing page with searchable article listing
- All HTML blog articles (33 technical articles)

### Manual Deployment

You can also trigger a manual deployment:
- Go to **Actions** tab in your GitHub repository
- Select "Deploy to GitHub Pages" workflow
- Click "Run workflow"

### Local Development

To preview the site locally:
```bash
cd poster
python3 -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions workflow
├── poster/
│   ├── index.html              # Landing page
│   └── *.html                  # Technical articles
└── README.md
```

## Features

- **Searchable Interface**: Filter articles by title or tags
- **Category Filters**: Energy, Mining, Infrastructure, Machine Learning
- **Responsive Design**: Works on desktop and mobile devices
- **Automatic Deployment**: Updates automatically on push to main branch

## Technologies

- Pure HTML/CSS/JavaScript (no build step required)
- GitHub Actions for CI/CD
- GitHub Pages for hosting
