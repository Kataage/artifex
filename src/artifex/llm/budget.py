from __future__ import annotations


class ContextBudgetExceeded(RuntimeError):
    def __init__(
        self,
        *,
        input_tokens: int,
        max_input_tokens: int,
        context_window_tokens: int,
        reserved_output_tokens: int,
        repair_headroom_tokens: int,
    ) -> None:
        super().__init__(
            "LLM input context exceeds configured budget: "
            f"{input_tokens} > {max_input_tokens} input tokens "
            f"(context={context_window_tokens}, "
            f"output_reserve={reserved_output_tokens}, "
            f"repair_headroom={repair_headroom_tokens})"
        )
        self.input_tokens = input_tokens
        self.max_input_tokens = max_input_tokens
        self.context_window_tokens = context_window_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.repair_headroom_tokens = repair_headroom_tokens
