#!/usr/bin/env python
#!/usr/bin/env python
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit import PromptSession
import asyncio

class TerminalTUI:
    def __init__(self):
        # Left pane: logs
        self.left_pane = TextArea(
            text="System Logs:\n",
            scrollbar=True,
            focusable=False,
            read_only=True,
        )

        # Right top pane: output
        self.output_pane = TextArea(
            scrollbar=True,
            multiline=True,
            read_only=True,
        )

        # History for PromptSession
        self.history = InMemoryHistory()
        self.session = PromptSession(history=self.history, auto_suggest=AutoSuggestFromHistory())

        # Right bottom placeholder window
        self.input_placeholder = Window(height=1, char="─", style="class:line")

        # Layout
        self.root_container = VSplit(
            [
                self.left_pane,
                Window(width=1, char="│", style="class:line"),
                HSplit(
                    [
                        self.output_pane,
                        self.input_placeholder,  # visual separator
                    ]
                ),
            ]
        )

        self.layout = Layout(self.root_container)
        self.kb = KeyBindings()

        @self.kb.add("c-c")
        def _(event):
            event.app.exit()

        self.app = Application(layout=self.layout, key_bindings=self.kb, full_screen=True, refresh_interval=0.1)

    async def run(self):
        # Run the Application in the background
        app_task = asyncio.create_task(self.app.run_async())

        # Input loop using PromptSession
        while True:
            try:
                text = await self.session.prompt_async("> ")
                if text.strip():
                    self.output_pane.buffer.insert_text(f"> {text}\n", move_cursor=True)
            except (EOFError, KeyboardInterrupt):
                break

        app_task.cancel()


if __name__ == "__main__":
    asyncio.run(TerminalTUI().run())

