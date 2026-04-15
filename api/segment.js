// POST /api/segment — Submit segmentation job to RunPod
// Body: { input: { image, cloth_type } }
// Returns: { id }

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' });

  const ep = process.env.SEG_EP;
  const key = process.env.SEG_KEY;
  if (!ep || !key) return res.status(500).json({ error: 'Segmentation not configured' });

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
