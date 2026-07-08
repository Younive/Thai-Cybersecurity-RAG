import { test, expect, type Page, type Route } from "@playwright/test";

// --- mock payloads -------------------------------------------------------

const SUCCESS = {
  answer: "The top risk is Broken Access Control [Source: owasp-top-10, Page 12].",
  sources: [
    { source: "owasp-top-10.pdf", page: 12, preview: "A01:2021 Broken Access Control" },
    { source: "mitre-attack.pdf", page: 3, preview: "Adversaries maintain access" },
  ],
  citations: [],
};

const EMPTY = { answer: "No relevant documents found. Try rephrasing or raising k.", sources: [], citations: [] };

/** Fulfill /api/query with a fixed JSON body. */
function mockQuery(page: Page, body: unknown, status = 200) {
  return page.route("**/api/query", (route: Route) =>
    route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) })
  );
}

async function ask(page: Page, question: string) {
  await page.getByLabel("Ask a security question").fill(question);
  await page.getByLabel("Ask a security question").press("Enter");
}

// --- tests ---------------------------------------------------------------

test("empty state: samples, corpus tags, disabled send", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(".sample")).toHaveCount(3);
  await expect(page.locator(".corpus__tag")).toHaveCount(3);
  await expect(page.locator(".composer__send")).toBeDisabled();
});

test("happy path: answer, citation chip, evidence cards", async ({ page }) => {
  await mockQuery(page, SUCCESS);
  await page.goto("/");
  await ask(page, "top owasp risk?");

  await expect(page.locator(".turn--user .turn__q")).toHaveText("top owasp risk?");
  await expect(page.locator("article.report .report__head")).toHaveText("Answer");
  await expect(page.locator("span.cite")).toContainText("OWASP");
  await expect(page.locator(".evidence__label")).toHaveText("Retrieved evidence (2)");
  await expect(page.locator(".evidence .card")).toHaveCount(2);
});

test("loading indicator shows while request is in flight", async ({ page }) => {
  await page.route("**/api/query", async (route: Route) => {
    await new Promise((r) => setTimeout(r, 500));
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(SUCCESS) });
  });
  await page.goto("/");
  await ask(page, "q");
  await expect(page.locator(".thinking")).toBeVisible();
  await expect(page.locator("article.report")).toBeVisible(); // resolved
  await expect(page.locator(".thinking")).toHaveCount(0);
});

test("sample button sends its preset question", async ({ page }) => {
  await mockQuery(page, SUCCESS);
  await page.goto("/");
  await page.locator(".sample").first().click();
  await expect(page.locator(".turn--user .turn__q")).toContainText("Broken Access Control");
});

test("HTTP 502 renders an error bubble", async ({ page }) => {
  await mockQuery(page, { detail: "Retrieval failed" }, 502);
  await page.goto("/");
  await ask(page, "q");
  const err = page.locator("article.report--error");
  await expect(err).toHaveAttribute("role", "alert");
  await expect(err.locator(".report__head")).toHaveText("Error");
});

test("malformed response renders an error bubble", async ({ page }) => {
  await mockQuery(page, { answer: 123 }); // answer not a string -> parse throws
  await page.goto("/");
  await ask(page, "q");
  await expect(page.locator("article.report--error")).toContainText("Malformed response from backend.");
});

test("empty results: answer shown, no evidence section", async ({ page }) => {
  await mockQuery(page, EMPTY);
  await page.goto("/");
  await ask(page, "q");
  await expect(page.locator(".report__body")).toContainText("No relevant documents found");
  await expect(page.locator(".evidence")).toHaveCount(0);
});

test("k slider value is sent in the request body", async ({ page }) => {
  let sentK: number | undefined;
  await page.route("**/api/query", (route: Route) => {
    sentK = route.request().postDataJSON().k;
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(EMPTY) });
  });
  await page.goto("/");
  await page.locator("#k").fill("12");
  await ask(page, "q");
  await expect(page.locator(".report__body")).toBeVisible();
  expect(sentK).toBe(12);
});
