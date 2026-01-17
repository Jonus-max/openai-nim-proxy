// server.js - OpenAI to NVIDIA NIM API Proxy (TIMEOUT-RESISTANT)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Validate API key
if (!NIM_API_KEY) {
  console.error('❌ ERROR: NIM_API_KEY environment variable is not set!');
  process.exit(1);
}

// Configuration
const SHOW_REASONING = false;
const ENABLE_THINKING_MODE = false;
const MAX_CACHE_SIZE = 100;
const FORCE_STREAMING = true; // 🔥 NEW: Force streaming to prevent timeouts

// Model cache
const validatedModels = new Map();

// Model mapping
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking'
};

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    forced_streaming: FORCE_STREAMING,
    cached_models: validatedModels.size
  });
});

// List models
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Optimized model selection
function getOptimalModel(requestedModel) {
  if (MODEL_MAPPING[requestedModel]) {
    return MODEL_MAPPING[requestedModel];
  }
  
  if (validatedModels.has(requestedModel)) {
    return validatedModels.get(requestedModel);
  }
  
  const modelLower = requestedModel.toLowerCase();
  let fallbackModel;
  
  if (modelLower.includes('deepseek')) {
    fallbackModel = 'deepseek-ai/deepseek-v3.1';
  } else if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
    fallbackModel = 'meta/llama-3.1-405b-instruct';
  } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
    fallbackModel = 'meta/llama-3.1-70b-instruct';
  } else {
    fallbackModel = 'deepseek-ai/deepseek-v3.1';
  }
  
  if (validatedModels.size >= MAX_CACHE_SIZE) {
    const firstKey = validatedModels.keys().next().value;
    validatedModels.delete(firstKey);
  }
  validatedModels.set(requestedModel, fallbackModel);
  
  return fallbackModel;
}

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Validate required fields
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: {
          message: 'messages field is required and must be a non-empty array',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }
    
    const nimModel = getOptimalModel(model || 'gpt-4o');
    
    // 🔥 FORCE STREAMING to prevent timeouts
    const useStreaming = FORCE_STREAMING || stream;
    
    console.log(`📨 Request: ${model || 'default'} -> ${nimModel} (streaming: ${useStreaming})`);
    
    // Build request
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 2048, // Reduced for faster responses
      stream: useStreaming
    };
    
    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }
    
    const startTime = Date.now();
    
    // Make API request
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: useStreaming ? 'stream' : 'json',
      timeout: 180000, // 3 minutes (increased from 2)
      maxRedirects: 5
    });
    
    if (useStreaming) {
      // Set headers immediately
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no'); // 🔥 Disable buffering for immediate streaming
      
      // Send a heartbeat to keep connection alive
      const heartbeat = setInterval(() => {
        res.write(': heartbeat\n\n');
      }, 15000); // Every 15 seconds
      
      let buffer = '';
      let reasoningStarted = false;
      let firstChunkReceived = false;
      
      response.data.on('data', (chunk) => {
        if (!firstChunkReceived) {
          console.log(`⚡ First chunk received: ${Date.now() - startTime}ms`);
          firstChunkReceived = true;
        }
        
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  data.choices[0].delta.content = content || '';
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              console.error('Parse error:', e.message);
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => {
        clearInterval(heartbeat);
        console.log(`✅ Stream completed: ${Date.now() - startTime}ms`);
        res.end();
      });
      
      response.data.on('error', (err) => {
        clearInterval(heartbeat);
        console.error('❌ Stream error:', err);
        res.end();
      });
      
      // Handle client disconnect
      req.on('close', () => {
        clearInterval(heartbeat);
        console.log('Client disconnected');
        response.data.destroy();
      });
      
    } else {
      // Non-streaming response (only if FORCE_STREAMING is false)
      console.log(`⚡ Response time: ${Date.now() - startTime}ms`);
      
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          return {
            index: choice.index,
            message: {
              role: choice.message.role,
              content: fullContent
            },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('❌ Proxy error:', error.message);
    if (error.code === 'ECONNABORTED') {
      console.error('Request timed out after 3 minutes');
    }
    if (error.response?.data) {
      console.error('API error details:', error.response.data);
    }
    
    // Don't try to send error if headers already sent (streaming case)
    if (!res.headersSent) {
      res.status(error.response?.status || 500).json({
        error: {
          message: error.message || 'Internal server error',
          type: 'invalid_request_error',
          code: error.response?.status || 500
        }
      });
    }
  }
});

// Catch-all
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`✅ OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`🏥 Health check: http://localhost:${PORT}/health`);
  console.log(`💭 Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`🧠 Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`🌊 Forced streaming: ${FORCE_STREAMING ? 'ENABLED (recommended)' : 'DISABLED'}`);
});
