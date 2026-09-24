import asyncio
import base64
from typing import Tuple
from playwright.async_api import async_playwright, Page, Browser


class MultimodalVisionBrowser:
    """
    Persistent Playwright browser for UX evaluation.

    A single instance is started once per run and reused across every action,
    so navigation/clicks actually persist (unlike the old per-action restart).
    Each element it reports carries an `index` and its on-screen rectangle, so
    the agent can act by index and we click by real coordinates -- this avoids
    brittle CSS/text selectors that used to time out for 30s.
    """

    def __init__(self, headless: bool = False):
        self.headless = headless
        self._playwright = None
        self._browser: Browser = None
        self._context = None
        self._page: Page = None

    async def start(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(viewport={"width": 1280, "height": 800})
        self._page = await self._context.new_page()
        # If a click opens a new tab, follow it instead of getting stranded.
        self._context.on("page", self._on_new_page)
        # Keep individual operations short so a bad target never stalls the run.
        self._page.set_default_timeout(6000)

    def _on_new_page(self, page):
        self._page = page

    def _active_page(self) -> Page:
        """Return a live page, recovering if the current one was closed."""
        if self._page and not self._page.is_closed():
            return self._page
        open_pages = [p for p in self._context.pages if not p.is_closed()]
        if open_pages:
            self._page = open_pages[-1]
        return self._page

    async def stop(self):
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass

    @property
    def current_url(self) -> str:
        try:
            return self._active_page().url
        except Exception:
            return ""

    async def navigate(self, url: str) -> Tuple[str, list, str]:
        """Navigate to a URL and capture the multimodal state."""
        if not url.startswith("http"):
            url = f"https://{url}"
        await self._active_page().goto(url, wait_until="domcontentloaded", timeout=30000)
        await self._settle()
        return await self.capture_multimodal_state()

    async def execute_action(
        self, action_type: str, index: int = None, selector: str = None, value: str = None
    ) -> Tuple[str, list, str]:
        """
        Execute one action on the *current* page and capture the resulting state.
        Prefers clicking by element index (coordinate click); falls back to a
        selector if provided.
        """
        try:
            page = self._active_page()
            target = await self._resolve_element(index) if index is not None else None

            if action_type == "click":
                if target:
                    await self._click_xy(target)
                elif selector:
                    await page.click(selector, delay=80)

            elif action_type == "type":
                if target:
                    await self._click_xy(target)
                elif selector:
                    await page.click(selector)
                if value:
                    await self._active_page().keyboard.type(value, delay=30)

            elif action_type == "scroll":
                await page.mouse.wheel(0, 700)

            await self._settle()
        except Exception as e:
            # Surface the error in the DOM-less capture but never crash the run.
            print(f"[browser] action '{action_type}' failed: {e}")
        return await self.capture_multimodal_state()

    async def _resolve_element(self, index: int):
        """Re-read the current interactive elements and return the one at `index`."""
        elements = await self._extract_elements()
        if 0 <= index < len(elements):
            return elements[index]
        return None

    async def _click_xy(self, el: dict):
        """Click the visual centre of an element; scroll it into view first."""
        page = self._active_page()
        cx = el["x"] + el["width"] / 2
        cy = el["y"] + el["height"] / 2
        # If the element sits below the fold, nudge the page so it's clickable.
        if cy > 780:
            await page.mouse.wheel(0, cy - 400)
            await asyncio.sleep(0.3)
            cy = 400
        await page.mouse.move(cx, cy)
        await asyncio.sleep(0.2)
        await page.mouse.click(cx, cy, delay=80)

    async def _settle(self):
        """Give the page a moment to react without hanging on networkidle forever."""
        try:
            await self._active_page().wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        await asyncio.sleep(0.6)

    async def _extract_elements(self) -> list:
        js_extraction_script = """
        () => {
            const nodes = document.querySelectorAll(
                'button, a, input, select, textarea, [role="button"]'
            );
            const data = [];
            let i = 0;
            for (let el of nodes) {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                const visible = rect.width > 4 && rect.height > 4 &&
                    rect.top >= 0 && rect.left >= 0 &&
                    rect.top < window.innerHeight &&
                    style.visibility !== 'hidden' && style.display !== 'none';
                if (!visible) continue;
                data.push({
                    index: i++,
                    tag: el.tagName.toLowerCase(),
                    text: (el.innerText || el.value || '').trim().substring(0, 60),
                    placeholder: el.placeholder || '',
                    aria_label: el.getAttribute('aria-label') || '',
                    id: el.id || '',
                    x: Math.round(rect.x),
                    y: Math.round(rect.y),
                    width: Math.round(rect.width),
                    height: Math.round(rect.height)
                });
            }
            return data;
        }
        """
        return await self._active_page().evaluate(js_extraction_script)

    async def capture_multimodal_state(self) -> Tuple[str, list, str]:
        """Capture a screenshot + simplified interactive-element list + title.

        Fully defensive: a mid-navigation page or a closed tab yields empty
        results rather than crashing the whole run.
        """
        page = self._active_page()
        screenshot_b64, elements, title = "", [], ""
        try:
            screenshot_bytes = await page.screenshot(full_page=False)
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        except Exception as e:
            print(f"[browser] screenshot failed: {e}")
        try:
            elements = await self._extract_elements()
        except Exception as e:
            print(f"[browser] element extraction failed: {e}")
        try:
            title = await page.title()
        except Exception:
            pass
        return screenshot_b64, elements, title
