# Deploying SuperTeacher Backend to Render

This guide provides step-by-step instructions for deploying the SuperTeacher backend to Render with Firebase credentials properly configured.

## Prerequisites

- A [Render](https://render.com) account
- A Firebase project with Storage enabled
- Your `firebase-service-account.json` file

## Step 1: Extract Firebase Credentials

First, you need to extract the correct credentials from your Firebase service account file.

1. Open your `firebase-service-account.json` file
2. Note the following values:
   - `project_id`
   - `client_email`
   - `private_key` (this is the long string starting with "-----BEGIN PRIVATE KEY-----")
   - `client_id` (optional)

## Step 2: Format the Private Key

The private key needs special handling for environment variables:

1. Copy the entire private key, including the BEGIN and END markers
2. Make sure all newlines are represented as `\n`
3. **Do not** add quotes around the key in Render's environment variables

For example, if your key in the JSON looks like:

```
"private_key": "-----BEGIN PRIVATE KEY-----\nABC123...\n...XYZ\n-----END PRIVATE KEY-----\n"
```

Then in Render's environment variables, it should be:

```
FIREBASE_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----\nABC123...\n...XYZ\n-----END PRIVATE KEY-----\n
```

## Step 3: Create a Web Service on Render

1. Log in to your [Render Dashboard](https://dashboard.render.com)
2. Click "New +" and select "Web Service"
3. Connect your Git repository
4. Configure the service:
   - **Name**: `superteacher-backend` (or your preferred name)
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your deployment branch)
   - **Root Directory**: Leave blank or specify if in a subdirectory
   - **Runtime**: `Node`
   - **Build Command**: `npm install && npm run build`
   - **Start Command**: `npm start` (or `node dist/server.js`)
   - **Plan**: Select appropriate plan (Free tier for testing)

## Step 4: Configure Environment Variables

Click on "Advanced" and add the following Environment Variables:

```
NODE_ENV=production
PORT=10000

# Firebase Configuration
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_CLIENT_EMAIL=your-client-email@yourproject.iam.gserviceaccount.com
FIREBASE_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----\nYour-entire-private-key-with-newlines-as-\n\n-----END PRIVATE KEY-----\n
FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com

# OCR Configuration
OCR_ENDPOINT=https://grading-api.onrender.com/extract-text
OCR_DEMO_MODE=false
OCR_MAX_RETRIES=3
OCR_RETRY_DELAY=2000

# Other app-specific variables
# ...
```

**CRITICAL**: The private key must have `\n` for newlines and must NOT be wrapped in extra quotes.

## Step 5: Deploy Service

1. Click "Create Web Service"
2. Wait for the initial deployment to complete

## Step 6: Verify Deployment

1. Once deployed, check the logs for:

   - "Using Firebase credentials from environment variables"
   - "Firebase Admin initialized successfully"
   - "Firebase Storage bucket obtained successfully"

2. If you see errors like `DECODER routines::unsupported`, there's an issue with your private key format. Try:
   - Removing quotes that may be wrapping the key
   - Ensuring all newlines are properly formatted as `\n`
   - If needed, regenerate your service account key in Firebase

## Step 7: Update Client Configuration

Update your client app to use the new backend URL:

```
NEXT_PUBLIC_API_URL=https://your-service-name.onrender.com
```

## Troubleshooting

### Private Key Errors

If you see errors related to the private key:

1. Check for proper newline formatting using `\n`
2. Ensure the key isn't wrapped in quotes in Render's environment variables
3. Try regenerating the service account key in Firebase Console
4. If all else fails, you can temporarily encode the key in base64:
   ```
   // In your Firebase config
   const privateKey = Buffer.from(
     process.env.FIREBASE_PRIVATE_KEY_BASE64 || '',
     'base64'
   ).toString('utf8');
   ```

### Storage Bucket Issues

If storage operations fail but Firebase initializes:

1. Verify your bucket name is correct (usually `your-project-id.appspot.com`)
2. Check if the service account has the "Storage Admin" role in Firebase

### Permission Issues

If you see 403 errors:

1. Make sure your service account has the right permissions in Firebase
2. Verify that Firebase Storage rules allow the operations you're performing
