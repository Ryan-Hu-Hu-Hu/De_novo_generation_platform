# LINE Chatbot Setup Guide

## 1. Create a LINE Messaging API Channel

1. Go to [LINE Developers Console](https://developers.line.biz/console/)
2. Log in with your LINE account
3. Click **Create a new provider** (or select an existing one)
4. Click **Create a new channel** → select **Messaging API**
5. Fill in:
   - Channel type: Messaging API
   - Provider: (your provider)
   - Channel name: De Novo Protein Generator
   - Channel description: AI-powered de novo protein design assistant
   - Category / Subcategory: (any)
6. Click **Create**

## 2. Copy Credentials

In your new channel's **Basic settings** tab:
- Copy the **Channel Secret**

In the **Messaging API** tab:
- Scroll to **Channel access token** → click **Issue** → copy the token

## 3. Configure Webhook

In the **Messaging API** tab:
1. Under **Webhook settings**, enable **Use webhook**
2. Set **Webhook URL** to:
   ```
   https://graceng-ncku.com/_functions/lineWebhook
   ```
3. Click **Verify** to test the connection

## 4. Create `.env` File

In the project root (`De_novo_generation_platform/`), create a file named `.env`:

```env
LINE_CHANNEL_SECRET=your_channel_secret_here
LINE_CHANNEL_ACCESS_TOKEN=your_channel_access_token_here

# Optional
FLASK_DEBUG=0
```

## 5. Install Dependencies

```bash
conda activate lin
pip install -r requirements_chatbot.txt
```

## 6. Download External Tools

### P2Rank (active-site prediction)
```bash
mkdir -p tools/p2rank
# Download from https://github.com/rdk/p2rank/releases
wget https://github.com/rdk/p2rank/releases/download/2.4.2/p2rank_2.4.2.tar.gz -P tools/
tar -xzf tools/p2rank_2.4.2.tar.gz -C tools/p2rank/ --strip-components=1
chmod +x tools/p2rank/prank
# Requires Java 11+: sudo apt install openjdk-11-jre
```

### Seq2Topt (optimal temperature prediction)
```bash
git clone https://github.com/SizheQiu/Seq2Topt tools/Seq2Topt
cd tools/Seq2Topt
conda env create -f environment.yml  # or follow the repo's README
cd ../..
```

## 7. Start the Server

```bash
conda activate lin
module load cuda/12.8  # if on HPC with CUDA
python main.py
```

The server starts on port **5000** by default.

## 8. Expose Flask with Cloudflared Tunnel

The Wix webhook forwards LINE events to your Flask server, so Flask must be reachable from the internet.
Use **cloudflared** (free, no account needed for quick tunnels):

```bash
# Download cloudflared (run once)
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O ~/cloudflared
chmod +x ~/cloudflared

# In a separate terminal, start the tunnel while Flask is running:
~/cloudflared tunnel --url http://localhost:5000
```

Cloudflared will print a public URL like:
```
https://some-random-name.trycloudflare.com
```

Copy this URL — you will paste it into Wix in step 9.

> **Note:** The quick tunnel URL changes every time cloudflared restarts.
> For a permanent URL, create a free Cloudflare account and use a named tunnel.

## 9. Update Wix to Forward Events to Flask

In the Wix Editor, open **Backend → http-functions.js** and set `FLASK_BACKEND_URL` to
the cloudflared URL you got in step 8:

```js
const FLASK_BACKEND_URL = 'https://some-random-name.trycloudflare.com/callback';
```

The full `http-functions.js` should forward LINE events to Flask like this:

```js
import { ok, badRequest, serverError } from 'wix-http-functions';
import { createHmac } from 'crypto';

const LINE_CHANNEL_SECRET = 'YOUR_CHANNEL_SECRET';
const FLASK_BACKEND_URL   = 'https://some-random-name.trycloudflare.com/callback';

export async function post_lineWebhook(request) {
  try {
    const body      = await request.body.text();
    const signature = request.headers['x-line-signature'];

    // Validate LINE signature
    if (body && signature) {
      const hash = createHmac('SHA256', LINE_CHANNEL_SECRET)
        .update(body)
        .digest('base64');
      if (hash !== signature) {
        return badRequest({ body: 'Invalid signature' });
      }
    }

    // Forward to Flask backend
    await fetch(FLASK_BACKEND_URL, {
      method:  'POST',
      headers: {
        'Content-Type':    'application/json',
        'X-Line-Signature': signature || '',
      },
      body,
    });

    return ok({ body: JSON.stringify({ status: 'ok' }) });
  } catch (err) {
    return serverError({ body: JSON.stringify({ error: err.message }) });
  }
}

export function get_lineWebhook(request) {
  return ok({ body: JSON.stringify({ status: 'webhook active' }) });
}
```

After pasting the new URL, **publish** the Wix site.

## 9. Using the Chatbot

1. Find your bot in LINE (add via QR code in the **Messaging API** tab)
2. Send any message — the bot will ask for a PDB code
3. Enter a 4-letter PDB code (e.g. `1PMO`)
4. Select a reaction temperature from the quick-reply buttons (10–100 °C)
5. The bot will send progress updates while the pipeline runs
6. When complete, the best candidate sequence and its properties are returned

## Troubleshooting

| Problem | Solution |
|---|---|
| `Invalid LINE signature` | Check `LINE_CHANNEL_SECRET` in `.env` and in Wix `http-functions.js` |
| `No candidates found` | Increase `MAX_ITERATIONS` in `pipeline/config.py` |
| Flask not receiving messages | Check cloudflared is running and `FLASK_BACKEND_URL` in Wix is up to date |
| Cloudflared URL changed | Restart cloudflared, update `FLASK_BACKEND_URL` in Wix, republish site |
| P2Rank not found | Download binary to `tools/p2rank/prank` |
| Seq2Topt not found | Clone repo to `tools/Seq2Topt/` |
| CLEAN import error | Ensure `conda activate clean` has CLEAN installed |
