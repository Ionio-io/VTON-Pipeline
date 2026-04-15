// POST /api/runpod — Submit FASHN VTON job to RunPod
// Body: { input: { person_image, garment_image, category, ... } }
// Returns: { id }

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' });

  const ep = process.env.RP_EP;
  const key = process.env.RP_KEY;
  if (!ep || !key) return res.status(500).json({ error: 'RunPod VTON not configured' });

  try {
    const r = await fetch(`https://api.runpod.ai/v2/${ep}/run`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ input: req.body.input }),
    });

    if (!r.ok) return res.status(r.status).json({ error: await r.text() });
    res.status(200).json(await r.json());
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
}
