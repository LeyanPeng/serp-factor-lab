# Dashboard

Six pages over the results in `public/data/lab.json`, which is written by
`python/run_demo.py`. The dashboard computes no statistics of its own — if a
number here disagrees with the terminal, this file is stale and the fix is to
rerun the demo.

```bash
npm install
npm run dev        # http://localhost:3000
```

| route | what it answers |
|---|---|
| `/` | the 0.84-vs-0.73 question, both pipelines, side by side |
| `/calibration` | every factor's true weight against what was recovered |
| `/factors` | all 34 factors by evidence tier, plus the ones we refuse to track |
| `/clusters` | how much the weights move between intent clusters |
| `/simulator` | move a factor, see the rank change and its uncertainty band |
| `/experiments` | split tests, minimum detectable effect, interval coverage |

Next.js 16 App Router, TypeScript, Tailwind 4. No chart library — the error
bars and plots are hand-drawn SVG, because the error bar is the whole visual
argument and it was worth drawing properly.

`lib/lab.ts` holds the shapes and formatters and is client-safe; `lib/server.ts`
does the filesystem read and is marked `server-only`, so a client component
importing a type cannot drag `node:fs` into the browser bundle.
