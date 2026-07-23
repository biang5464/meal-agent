import { defineCloudflareConfig } from "@opennextjs/cloudflare";

// Minimal config — no ISR, no image optimisation, no middleware overrides.
// Secrets (MEAL_AGENT_BACKEND_URL, MEAL_AGENT_API_KEY) are injected at
// runtime via Cloudflare Secrets, not here.
export default defineCloudflareConfig();
