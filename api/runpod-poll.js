// GET /api/runpod-poll?id=xxx — Check RunPod VTON job status
// Returns: RunPod status response

export default async function handler(req, res) {
  if (req.method !== 'GET') return res.status(405).json({ error: 'GET only' });

  const ep = process.env.RP_EP;
  const key = process.env.RP_KEY;
  if (!ep || !key) return res.status(500).json({ error: 'RunPod VTON not configured' });

  const id = req.query.id;
  if (!id) return res.status(400).json({ error: 'Missing id' });

  try {
    const r = await fetch(`https://api.runpod.ai/v2/${ep}/status/${id}`, {
      headers: { 'Authorization': `Bearer ${key}` },
    });

    if (!r.ok) return res.status(r.status).json({ error: await r.text() });
    res.status(200).json(await r.json());
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
}
