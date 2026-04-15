// POST /api/fal-poll — Check fal.ai job status, return result if done
// Body: { status_url, response_url }
// Returns: { status } or full result if completed

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' });

  const key = process.env.FAL_KEY;
  if (!key) return res.status(500).json({ error: 'FAL_KEY not configured' });

  const { status_url, response_url } = req.body;
  if (!status_url || !response_url) return res.status(400).json({ error: 'Missing URLs' });

  // Security: only proxy to fal.ai
  if (!status_url.startsWith('https://queue.fal.run/') || !response_url.startsWith('https://queue.fal.run/')) {
    return res.status(400).json({ error: 'Invalid URL' });
  }

  try {
    const sr = await fetch(status_url, { headers: { 'Authorization': `Key ${key}` } });
    if (!sr.ok) return res.status(sr.status).json({ error: await sr.text() });
    const s = await sr.json();

    if (s.status === 'COMPLETED') {
      const rr = await fetch(response_url, { headers: { 'Authorization': `Key ${key}` } });
      if (!rr.ok) return res.status(rr.status).json({ error: await rr.text() });
      const result = await rr.json();
      return res.status(200).json({ status: 'COMPLETED', result });
    }

    res.status(200).json({ status: s.status });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
}
