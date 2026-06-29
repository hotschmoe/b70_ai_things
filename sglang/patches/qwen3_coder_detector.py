import ast
import json
import logging
import re
from typing import Any, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

# =============================================================================
# B70-STREAM-PATCH (2026-06-29)
# -----------------------------------------------------------------------------
# Baked sglang qwen3_coder streaming parser BUFFERS each <parameter=name>VALUE
# </parameter> body and emits nothing until the closing </parameter> arrives
# (see upstream: vLLM issue #30439, sglang main has the same behavior). For a
# large string param (a file written via a tool call) that is MINUTES of zero
# bytes to the client, which trips client idle timeouts (Pi/Hermes) mid-call ->
# empty tool args / "terminated".
#
# This file is a FAITHFUL COPY of the baked qwen3_coder_detector.py for sglang
# 0.5.6.post3.dev6841, with ONE behavioral change: STRING-typed parameter values
# now STREAM incrementally as JSON-string-content deltas, with end-token
# lookahead (hold back the last N chars so a terminator can never split across a
# delta). Non-string params and the null-literal special case fall back to the
# original one-shot path, so the concatenated `arguments` is byte-identical to
# the non-streaming result. Mounted over the baked file by serve.sh.
#
# Only __init__, parse_streaming_increment, and three small helpers differ from
# upstream; everything else is verbatim. Search this file for B70-STREAM-PATCH.
# =============================================================================


class Qwen3CoderDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()

        # Sentinel tokens
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"

        # Regex for non-streaming fallback
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        # Streaming State
        # Base class already initializes _buffer, we just use it directly
        # No need to check with hasattr - we control the lifecycle through inheritance

        # Index pointing to the next character to be processed in buffer
        self.parsed_pos: int = 0
        # Parameter count inside the current tool being processed, used to determine whether to add comma
        self.current_tool_param_count: int = 0
        # Flag indicating whether current tool has already sent '{'
        self.json_started: bool = False

        # [FIX] New state flag: mark whether inside tool_call structure block
        self.is_inside_tool_call: bool = False

        # Initialize attributes that were missing in the original PR
        self.current_func_name: Optional[str] = None

        # B70-STREAM-PATCH: incremental string-parameter streaming state.
        # _sp_active: currently streaming a string param value (opening quote
        #   already emitted, closing quote not yet). _sp_name: that param's name.
        # _term_holdback: chars to hold back from each incremental emit so a
        #   parameter terminator (or the trailing \n right before it) is never
        #   emitted as content -- = the longest terminator length.
        self._sp_active: bool = False
        self._sp_name: Optional[str] = None
        self._term_holdback: int = max(
            len(self.parameter_end_token),
            len(self.parameter_prefix),
            len(self.function_end_token),
        )

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def _get_arguments_config(
        self, func_name: str, tools: Optional[list[Tool]]
    ) -> dict:
        """Extract argument configuration for a function."""
        if tools is None:
            return {}
        for config in tools:
            try:
                config_type = config.type
                config_function = config.function
                config_function_name = config_function.name
            except AttributeError:
                continue

            if config_type == "function" and config_function_name == func_name:
                try:
                    params = config_function.parameters
                except AttributeError:
                    return {}

                if isinstance(params, dict) and "properties" in params:
                    return params["properties"]
                elif isinstance(params, dict):
                    return params
                else:
                    return {}
        logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
        return {}

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.warning(
                    f"Parsed parameter '{param_name}' is not defined in the tool "
                    f"parameters for tool '{func_name}', directly returning the string value."
                )
            return param_value

        if (
            isinstance(param_config[param_name], dict)
            and "type" in param_config[param_name]
        ):
            param_type = str(param_config[param_name]["type"]).strip().lower()
        else:
            param_type = "string"
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                param_value = int(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not an integer in tool "
                    f"'{func_name}', degenerating to string."
                )
            return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                maybe_convert = (
                    False if "." in param_value or "e" in param_value.lower() else True
                )
                param_value: float = float(param_value)
                if maybe_convert and param_value.is_integer():
                    param_value = int(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a float in tool "
                    f"'{func_name}', degenerating to string."
                )
            return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a boolean (`true` of `false`) in tool '{func_name}', degenerating to false."
                )
            return param_value == "true"
        else:
            if (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            ):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except Exception:
                    logger.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' cannot be parsed with json.loads in tool "
                        f"'{func_name}', will try other methods to parse it."
                    )
            try:
                param_value = ast.literal_eval(param_value)  # safer
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' cannot be converted via Python `ast.literal_eval()` in tool '{func_name}', degenerating to string."
                )
            return param_value

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-shot parsing for non-streaming scenarios."""
        if self.tool_call_start_token not in text:
            return StreamingParseResult(normal_text=text)

        calls = []
        try:
            # Simple cleanup of the text to find tool calls
            # Note: This is a simplified regex approach consistent with vLLM
            raw_tool_calls = self.tool_call_regex.findall(text)
            if not raw_tool_calls:
                # Fallback: maybe the whole text is inside the tag or tags are stripped
                if self.tool_call_prefix in text:
                    raw_tool_calls = [text]

            tool_idx = 0
            for tool_content in raw_tool_calls:
                # Find function calls
                funcs = self.tool_call_function_regex.findall(tool_content)
                for func_match in funcs:
                    func_body = func_match[0] or func_match[1]
                    if ">" not in func_body:
                        continue

                    name_end = func_body.index(">")
                    func_name = func_body[:name_end]
                    params_str = func_body[name_end + 1 :]

                    param_config = self._get_arguments_config(func_name, tools)
                    parsed_params = {}

                    for p_match in self.tool_call_parameter_regex.findall(params_str):
                        if ">" not in p_match:
                            continue
                        p_idx = p_match.index(">")
                        p_name = p_match[:p_idx]
                        p_val = p_match[p_idx + 1 :]
                        # Remove prefixing and trailing \n
                        if p_val.startswith("\n"):
                            p_val = p_val[1:]
                        if p_val.endswith("\n"):
                            p_val = p_val[:-1]

                        parsed_params[p_name] = self._convert_param_value(
                            p_val, p_name, param_config, func_name
                        )

                    calls.append(
                        ToolCallItem(
                            tool_index=tool_idx,
                            name=func_name,
                            parameters=json.dumps(parsed_params, ensure_ascii=False),
                        )
                    )
                    tool_idx += 1

            # Determine normal text (text before the first tool call)
            start_idx = text.find(self.tool_call_start_token)
            if start_idx == -1:
                start_idx = text.find(self.tool_call_prefix)
            normal_text = text[:start_idx] if start_idx > 0 else ""

            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    # -------------------------------------------------------------------------
    # B70-STREAM-PATCH helpers
    # -------------------------------------------------------------------------
    def _find_param_terminator(self, s: str):
        """Earliest parameter terminator in s. Returns (pos, token_len) or (-1, 0).

        Mirrors the candidate logic of the original one-shot parameter branch:
        a parameter ends at the nearest of </parameter> (consumed, len>0),
        the next <parameter= (not consumed), or </function> (not consumed).
        """
        candidates = []
        p = s.find(self.parameter_end_token)
        if p != -1:
            candidates.append((p, len(self.parameter_end_token)))
        p = s.find(self.parameter_prefix)
        if p != -1:
            candidates.append((p, 0))
        p = s.find(self.function_end_token)
        if p != -1:
            candidates.append((p, 0))
        if not candidates:
            return (-1, 0)
        return min(candidates, key=lambda x: x[0])

    def _is_string_param(self, param_name: str, tools: List[Tool]) -> bool:
        """True if the schema types this param as a string (or leaves it untyped,
        which _convert_param_value treats as a pass-through string)."""
        cfg = self._get_arguments_config(self.current_func_name, tools)
        spec = cfg.get(param_name) if isinstance(cfg, dict) else None
        if isinstance(spec, dict) and "type" in spec:
            t = str(spec["type"]).strip().lower()
            return t in ["string", "str", "text", "varchar", "char", "enum"]
        # Untyped/unknown -> _convert_param_value returns the raw string as-is.
        return True

    @staticmethod
    def _json_str_inner(s: str) -> str:
        """Escape s as JSON-string CONTENT (no surrounding quotes). JSON string
        escaping is per-character (ensure_ascii=False), so concatenating the
        inner-escapes of pieces equals the inner-escape of the concatenation --
        which is what makes incremental string streaming byte-exact."""
        return json.dumps(s, ensure_ascii=False)[1:-1]

    @staticmethod
    def _could_be_null(core: str) -> bool:
        """True while the value-so-far could still resolve to the literal `null`
        (which _convert_param_value maps to None / an UNQUOTED json null for ANY
        type). Until we can rule that out, we must NOT open a quoted string.
        The final value is null iff its content (after the leading-\\n strip and
        a single trailing-\\n strip) lowercases to "null"; so the danger set is
        the case-insensitive prefixes of "null\\n"."""
        target = "null\n"
        return len(core) <= len(target) and target.startswith(core.lower())

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Robust cursor-based streaming parser.

        B70-STREAM-PATCH: string-typed parameter values stream incrementally
        instead of buffering until </parameter>.
        """
        self._buffer += new_text

        # Guard against empty buffer
        if not self._buffer:
            return StreamingParseResult()

        calls = []
        normal_text_chunks = []

        while True:
            # Working text slice
            current_slice = self._buffer[self.parsed_pos :]

            # Optimization: If almost empty, wait for more
            if not current_slice:
                break

            # ---------------------------------------------------------------
            # B70-STREAM-PATCH: 0. Continuation of an in-flight string param.
            # We are between the opening quote and the closing quote of a
            # streamed string value; consume value content directly (this runs
            # BEFORE tag dispatch because current_slice now starts mid-value,
            # not at a tag).
            # ---------------------------------------------------------------
            if self._sp_active:
                s = current_slice
                term_pos, term_len = self._find_param_terminator(s)
                if term_pos != -1:
                    # Value complete: emit the remaining content (minus one
                    # trailing \n, matching the original cleanup), then close.
                    content = s[:term_pos]
                    if content.endswith("\n"):
                        content = content[:-1]
                    if content:
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                parameters=self._json_str_inner(content),
                            )
                        )
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id, parameters='"'
                        )
                    )
                    self.current_tool_param_count += 1
                    self.parsed_pos += term_pos + term_len
                    self._sp_active = False
                    self._sp_name = None
                    continue
                else:
                    # No terminator yet: emit everything except the last
                    # _term_holdback chars (which might be the start of a
                    # terminator or the to-be-stripped trailing \n).
                    if len(s) > self._term_holdback:
                        emit_piece = s[: len(s) - self._term_holdback]
                        if emit_piece:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    parameters=self._json_str_inner(emit_piece),
                                )
                            )
                            self.parsed_pos += len(emit_piece)
                    break  # wait for more text

            # -------------------------------------------------------
            # 1. Priority detection: check if it's the start of Tool Call
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_start_token):
                self.parsed_pos += len(self.tool_call_start_token)
                self.is_inside_tool_call = True
                continue

            # -------------------------------------------------------
            # 2. Function Name: <function=name>
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_prefix):
                end_angle = current_slice.find(">")
                if end_angle != -1:
                    func_name = current_slice[len(self.tool_call_prefix) : end_angle]

                    self.current_tool_id += 1
                    self.current_tool_name_sent = True
                    self.current_tool_param_count = 0
                    self.json_started = False
                    self.current_func_name = func_name

                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )

                    self.parsed_pos += end_angle + 1
                    continue
                else:
                    # Incomplete tag
                    break

            # -------------------------------------------------------
            # 3. Parameter: <parameter=name>value...
            # -------------------------------------------------------
            if current_slice.startswith(self.parameter_prefix):
                name_end = current_slice.find(">")
                if name_end == -1:
                    # Incomplete <parameter=...> tag
                    break

                param_name = current_slice[len(self.parameter_prefix) : name_end]
                value_region = current_slice[name_end + 1 :]
                term_pos, term_len = self._find_param_terminator(value_region)

                if term_pos != -1:
                    # ---- Full value present: ONE-SHOT (original behavior) ----
                    raw_value = value_region[:term_pos]

                    # Cleanup value
                    if raw_value.startswith("\n"):
                        raw_value = raw_value[1:]
                    if raw_value.endswith("\n"):
                        raw_value = raw_value[:-1]

                    # JSON Construction
                    if not self.json_started:
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id, parameters="{"
                            )
                        )
                        self.json_started = True

                    param_config = self._get_arguments_config(
                        self.current_func_name, tools
                    )
                    converted_val = self._convert_param_value(
                        raw_value, param_name, param_config, self.current_func_name
                    )

                    # Construct JSON fragment: "key": value
                    json_key_val = f"{json.dumps(param_name)}: {json.dumps(converted_val, ensure_ascii=False)}"

                    if self.current_tool_param_count > 0:
                        fragment = f", {json_key_val}"
                    else:
                        fragment = json_key_val

                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id, parameters=fragment
                        )
                    )
                    self.current_tool_param_count += 1

                    # Advance cursor
                    total_len = (name_end + 1) + term_pos + term_len
                    self.parsed_pos += total_len
                    continue

                # ---- No terminator yet (value still streaming) ----
                # B70-STREAM-PATCH: incrementally stream STRING params; fall back
                # to the original buffer-until-complete for non-string params and
                # for values that might still resolve to the literal null.
                lead_nl = value_region.startswith("\n")
                content_region = value_region[1:] if lead_nl else value_region

                if not self._is_string_param(param_name, tools):
                    break  # non-string (int/bool/object): wait for the full value
                # Only the part OUTSIDE the terminator-lookahead window is
                # guaranteed to be value content -- the last _term_holdback chars
                # may be a partial </parameter>. Decide the null-literal gate on
                # that guaranteed prefix only, so a value of exactly `null` is
                # never mis-streamed as the quoted string "null" (it must reach
                # the one-shot path and convert to an unquoted json null).
                guaranteed = content_region[
                    : max(0, len(content_region) - self._term_holdback)
                ]
                if self._could_be_null(guaranteed):
                    break  # might still become the literal `null`: wait

                # Open the JSON string for this param, then stream its content
                # via the _sp_active branch on subsequent loop iterations.
                if not self.json_started:
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters="{")
                    )
                    self.json_started = True

                key_open = f'{json.dumps(param_name)}: "'
                opening = f", {key_open}" if self.current_tool_param_count > 0 else key_open
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id, parameters=opening
                    )
                )

                # Advance past "<parameter=name>" and the single leading \n (if any);
                # the value content now begins at parsed_pos for the _sp_active branch.
                self.parsed_pos += (name_end + 1) + (1 if lead_nl else 0)
                self._sp_active = True
                self._sp_name = param_name
                continue

            # -------------------------------------------------------
            # 4. Function End: </function>
            # -------------------------------------------------------
            if current_slice.startswith(self.function_end_token):
                if not self.json_started:
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters="{")
                    )
                    self.json_started = True

                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters="}")
                )
                self.parsed_pos += len(self.function_end_token)
                self.current_func_name = None
                continue

            # -------------------------------------------------------
            # 5. Tool Call End: </tool_call>
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_end_token):
                self.parsed_pos += len(self.tool_call_end_token)
                self.is_inside_tool_call = False  # [FIX] Exit tool call region
                continue

            # -------------------------------------------------------
            # 6. Handling content / whitespace / normal text
            # -------------------------------------------------------
            # If current position is not the start of a tag (i.e., doesn't start with <), it might be plain text,
            # or a newline between two tags.
            # But we need to be careful not to output truncated tags like "<fun" as text.

            next_open_angle = current_slice.find("<")

            if next_open_angle == -1:
                # This entire segment is plain text
                if not self.is_inside_tool_call:
                    normal_text_chunks.append(current_slice)
                # [FIX] If inside tool call, discard this text (usually \n), don't append
                self.parsed_pos += len(current_slice)
                continue

            elif next_open_angle == 0:
                # Looks like a Tag, but doesn't match any known Tag above

                possible_tags = [
                    self.tool_call_start_token,
                    self.tool_call_end_token,
                    self.tool_call_prefix,
                    self.function_end_token,
                    self.parameter_prefix,
                    self.parameter_end_token,
                ]

                is_potential_tag = False
                for tag in possible_tags:
                    if tag.startswith(current_slice):
                        is_potential_tag = True
                        break

                if is_potential_tag:
                    break  # Wait for more
                else:
                    # Just a plain '<' symbol
                    if not self.is_inside_tool_call:
                        normal_text_chunks.append("<")
                    self.parsed_pos += 1
                    continue

            else:
                # '<' is in the middle
                text_segment = current_slice[:next_open_angle]
                if not self.is_inside_tool_call:
                    normal_text_chunks.append(text_segment)
                # [FIX] If inside tool call, discard whitespace/text before Tag
                self.parsed_pos += next_open_angle
                continue

        # Memory Cleanup: Slice the buffer
        # Keep unparsed part, discard parsed part
        if self.parsed_pos > 0:
            self._buffer = self._buffer[self.parsed_pos :]
            self.parsed_pos = 0

        normal_text = "".join(normal_text_chunks) if normal_text_chunks else ""
        return StreamingParseResult(calls=calls, normal_text=normal_text)

    def supports_structural_tag(self) -> bool:
        return True

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def get_structural_tag_name(self) -> str:
        return "qwen_3_coder"
