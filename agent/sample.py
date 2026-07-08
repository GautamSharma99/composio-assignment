"""The stratified verification sample (PRD §9).

20 apps chosen so the baseline is honest, not flattering:
  * >= 1 per category (all 10 covered),
  * every deliberately-gated app (DealCloud, PitchBook, Paygent, iPayX, fanbasis,
    Waterfall.io),
  * every disambiguation trap (Copper, Threads, Plain, Grain, Twenty),
  * plus easy-win contrasts (GitHub, Notion, Stripe, SendGrid, Firecrawl, OpenAI) so
    accuracy isn't measured only on the hard set.
"""

SAMPLE_IDS: list[int] = [
    1,   # GitHub        — Dev & Infra        — easy-win
    11,  # Notion        — Productivity       — easy-win
    20,  # Grain         — Productivity       — disambiguation trap
    30,  # Plain         — Comms              — disambiguation trap
    34,  # Copper        — CRM                — disambiguation trap
    37,  # Twenty        — CRM                — disambiguation trap
    41,  # Stripe        — Finance            — easy-win contrast
    43,  # PitchBook     — Finance            — gated
    44,  # DealCloud     — Finance            — gated
    45,  # Paygent       — Finance            — gated
    46,  # iPayX         — Finance            — gated
    52,  # SendGrid      — Marketing          — easy-win
    54,  # Meta Ads      — Marketing          — app-review gated
    65,  # Amazon SP-API — Commerce           — approval gated
    70,  # fanbasis      — Commerce           — gated
    73,  # Firecrawl     — Data/SEO/Scraping  — easy-win
    79,  # Waterfall.io  — Data/SEO/Scraping  — gated
    81,  # OpenAI        — AI & Research       — easy-win
    86,  # Devin         — AI & Research       — gated / waitlist
    91,  # Threads       — Social & Media      — disambiguation trap
]

SAMPLE_SET = set(SAMPLE_IDS)
