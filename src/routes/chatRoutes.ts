import { Router } from 'express';
import { chatController } from '../controllers/chatController';

const router = Router();

// Chat endpoint - handles both text messages and file uploads
router.post('/chat', chatController.handleChatRequest);

export default router;