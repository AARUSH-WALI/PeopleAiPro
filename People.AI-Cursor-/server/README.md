# Resume Parser API

A Node.js API for parsing and managing resumes and interviews, built with Express and Supabase.

## Features

- Upload and parse resumes
- Store and manage resume data
- Schedule and manage interviews
- Search and filter resumes
- File storage with Supabase Storage

## Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Supabase account

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Copy `.env.example` to `.env` and update with your Supabase credentials
4. Start the development server:
   ```bash
   npm run dev
   ```

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
PORT=5000
NODE_ENV=development
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_KEY=your_supabase_service_key
MAX_FILE_SIZE=10485760
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5000

# SMTP (for sending emails via Nodemailer)
# Preferred generic variables:
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_ADDRESS=your_gmail_address@gmail.com
EMAIL_PASSWORD=your_gmail_app_password
# Optional custom from name/address (falls back to EMAIL_ADDRESS)
EMAIL_FROM=People AI <your_gmail_address@gmail.com>

# Backwards-compat (optional): if set, these will also work
# GMAIL_USER=your_gmail_address@gmail.com
# GMAIL_APP_PASSWORD=your_gmail_app_password
```

## API Endpoints

### Resumes

- `GET /api/resumes` - Get all resumes (paginated)
- `GET /api/resumes/search?query=search_term` - Search resumes
- `GET /api/resumes/:id` - Get a specific resume
- `POST /api/resumes` - Create a new resume (with file upload)
- `PUT /api/resumes/:id` - Update a resume
- `DELETE /api/resumes/:id` - Delete a resume

### Interviews

#### Example with file upload (using form-data):
```bash
curl -X POST http://localhost:5000/api/resume \
  -H "Content-Type: multipart/form-data" \
  -F "resumeFile=@/path/to/resume.pdf" \
  -F 'data={"personalInfo":{"name":"John Doe"},...}'
```

#### Example with JSON only:
```bash
curl -X POST http://localhost:5000/api/resume \
  -H "Content-Type: application/json" \
  -d @parse.json
```

Supported file types: PDF, DOC, DOCX (max 5MB)

### Get Resume by ID

**GET** `/api/resume/:id`

Retrieve a stored resume by its ID, including file information if available.

### Get Resume File

**GET** `/api/resume/file/:id`

Redirects to the direct download URL of the resume file (if available).

### Search Resumes

**GET** `/api/resumes/search?query=search+term`

Search resumes by name, email, or best fit position.

## Project Structure

```
server/
├── .env.example           # Example environment variables
├── .gitignore
├── package.json
├── README.md
├── db-init.js             # Database initialization script
├── database.sql           # Database schema
└── index.js               # Main application file
```

## Deployment

1. Set the environment variables in your production environment
2. Install dependencies with `--production` flag:
   ```bash
   npm install --production
   ```
3. Start the server:
   ```bash
   npm start
   ```

For production, consider using a process manager like PM2:

```bash
npm install -g pm2
pm2 start index.js --name "resume-parser"
```

## Troubleshooting

### Database Initialization Issues

1. **Missing Service Role Key**: Ensure you've set the `SUPABASE_SERVICE_KEY` in your `.env` file.
2. **Connection Issues**: Verify your Supabase project URL and keys are correct.
3. **Schema Already Exists**: If tables already exist, the initialization will skip creating them.

### File Upload Issues

1. **File Size Limit**: The default limit is 5MB. You can adjust this in `index.js`.
2. **File Type**: Only PDF, DOC, and DOCX files are allowed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Supabase](https://supabase.com/) for the awesome backend-as-a-service
- [Express](https://expressjs.com/) for the web framework
- [Multer](https://github.com/expressjs/multer) for file uploads

## Database Schema

The database consists of the following tables:

- `personal_info`: Stores basic personal information
- `education`: Stores educational background
- `experience`: Stores work experience
- `skills`: Stores skills
- `achievements`: Stores achievements
- `projects`: Stores projects
- `resume_metadata`: Stores additional metadata and analytics

## Error Handling

The API returns appropriate HTTP status codes and JSON error messages in case of errors.

## License

MIT
