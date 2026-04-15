// POST /api/fal — Submit a job to fal.ai
// Body: { endpoint: "fal-ai/cat-vton", body: { ... } }
// Returns: { request_id, status_url, response_url }

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' });

  const key = process.env.FAL_KEY;
  if (!key) return res.status(500).json({ error: 'FAL_KEY not configured' });

  const { endpoint, body } = req.body;
  if (!endpoint || !body) return res.status(400).json({ error: 'Missing endpoint or body' });

  try {
    const r = await fetch(`https://queue.fal.run/${endpoint}`, {
      method: 'POST',
      headers: { 'Authorization': `Key ${key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!r.ok) {
      const t = await r.text();
      return res.status(r.status).json({ error: t });
    }

    const data = await r.json();
    res.status(200).json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
}
