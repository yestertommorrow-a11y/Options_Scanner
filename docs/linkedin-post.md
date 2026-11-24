# LinkedIn post: Options Scanner

## Post text (about 150 words)

Most options tools hand you a number and move on. I wanted one that shows its working.

So I built Options Scanner, an open-source tool that scans a US stock or ETF option chain for unusual volume, rich or cheap pricing, tight spreads and delta groups, and explains every figure it shows:

🔍 Click any contract and see its mid price, delta, theta, vega and edge worked out step by step, with the real numbers in each formula
📈 Eight interactive charts that turn the maths into pictures: where the price could land at expiry, the volatility smile, how time decay speeds up, how implied volatility sets the price
🎓 Beginner view (plain-English hints) and Pro view (dense trader layout), plus sunny and dark modes
🧪 Scan logic tested on hundreds of simulated option chains

Python (FastAPI + yfinance) and plain JavaScript charts, with a Tkinter desktop version too. Built with Claude Code.

It uses free, delayed Yahoo data, so it's for learning, not trading advice.

Code: https://github.com/yestertommorrow-a11y/Options_Scanner

What would you want an options tool to explain better?

#options #optionstrading #python #fintech #opensource #datavisualization

## Images (attach in this order; PNGs are 2x resolution)

1. `screenshots/1-market-read-dark.png`: hero image. Expected-move curve, IV vs realised vol, volatility smile, activity by strike.
2. `screenshots/2-contract-charts-sunny.png`: the four concept charts for one contract (payoff, time decay, odds, IV and price).
3. `screenshots/3-step-by-step-working-sunny.png`: the 13-step working behind one contract.
4. `screenshots/4-expanded-chart-sunny.png`: one chart expanded full-screen.
5. `screenshots/5-pro-view-dark.png`: Pro view.

`0-overview-dark.png` shows the summary cards and the first two charts, if you want a lighter single image.

Alt text for image 1: "Dark-mode dashboard of SPY options showing a probability curve of the expected price move, implied versus realised volatility, a volatility smile and volume versus open interest by strike."

## Tips

- LinkedIn shows the first ~2 lines before "see more", so the opening sentence carries the post.
- Post the 5 images as one multi-image post; image 1 is what people see first.
- The repo link in the post text renders as a preview card only if no images are attached, so keep the link in the text and use images for the visual hook.
- Pin your own first comment with the quick-start line: `python -m uvicorn main:app`, then open http://127.0.0.1:8000.
