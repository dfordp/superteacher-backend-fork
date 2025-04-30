# SuperTeacher Backend

Intelligent chatbot-style grading system for educators to process and assess student work.

## Image Processing Workflow

The backend now supports two methods of processing images:

1. **Direct File Upload**: The client uploads a photo to the backend, which is then:

   - Uploaded to Firebase Storage to generate a URL
   - Sent to the OCR service with `image_url` parameter to extract text
   - Processed by the grading system

2. **Image URL Processing**: You can directly send an image URL to the backend, and it will:
   - Skip the upload step and use the provided URL
   - Send to the OCR service with `image_url` parameter
   - Process the extracted text as usual

## API Endpoints

### POST /api/chat

This endpoint handles both text messages and image processing:

#### Text Message:

```json
{
  "message": "Your text message",
  "userId": "optional-user-id"
}
```

#### Process Image URL:

```json
{
  "image_url": "https://example.com/image.jpg",
  "userId": "optional-user-id"
}
```

#### File Upload:

Use multipart/form-data with:

- `image`: The image file
- `userId`: (Optional) user identifier
- `message`: (Optional) accompanying text message

## Testing with Postman

### Testing Direct Image Upload

1. Create a new POST request to `http://localhost:3001/api/chat`
2. Select the "Body" tab and choose "form-data"
3. Add the following keys:
   - Key: `image` (type: File) - select an image file from your computer
   - Key: `userId` (type: Text) - enter any string ID like "test-user-123" (optional)
4. Click "Send" to upload the image and receive the OCR results

### Testing with Image URL

1. Create a new POST request to `http://localhost:3001/api/chat`
2. Select the "Body" tab and choose "raw" with "JSON" format
3. Enter the following JSON:
   ```json
   {
     "image_url": "https://example.com/your-image.jpg",
     "userId": "test-user-123"
   }
   ```
4. Click "Send" to process the image URL directly

### Testing the OCR Endpoint Directly

If you need to test the OCR endpoint directly:

1. Create a new POST request to `https://grading-api.onrender.com/extract-text`
2. Select the "Body" tab and choose "form-data"
3. Add a key: `image_url` (type: Text) with the URL of your image
4. Click "Send" to verify the OCR extraction independently

**Note**: The OCR service expects the `image_url` parameter in form-data format, not as JSON.

## Environment Configuration

Key environment variables:

```
# Server and API settings
NODE_ENV=development
OPENAI_API_KEY=your-openai-key

# OCR Configuration
OCR_ENDPOINT=https://grading-api.onrender.com/extract-text

# Firebase Storage configuration
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_CLIENT_EMAIL=your-client-email@project.iam.gserviceaccount.com
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com

# Set to true to use mock storage without Firebase credentials
USE_MOCK_STORAGE=true
```

## Development

1. Install dependencies:

   ```
   npm install
   ```

2. Start the development server:

   ```
   npm run dev
   ```

3. For local testing without Firebase, set `USE_MOCK_STORAGE=true` in your `.env` file
