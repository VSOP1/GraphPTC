"""Run MCPMark's Notion login helper with a modern Notion navigation seam."""

from playwright.sync_api import Page

from src.mcp_services.notion.notion_login_helper import main


_page_goto = Page.goto


def _goto_without_full_load(self, url, **kwargs):
    if kwargs.get("wait_until") == "load":
        kwargs["wait_until"] = "domcontentloaded"
    return _page_goto(self, url, **kwargs)


Page.goto = _goto_without_full_load


if __name__ == "__main__":
    main()
