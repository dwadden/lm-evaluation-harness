import re
import json
from collections import Counter

from lm_eval.api.filter import Filter


class RegexFilter(Filter):
    """ """

    def __init__(
        self, regex_pattern: str = r"#### (\-?[0-9\.\,]+)", fallback: str = "[invalid]"
    ) -> None:
        """
        pass a string `regex` to run `re.compile(r"regex")` on.
        `fallback` defines the output returned if no matches for the regex are located.
        """
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.fallback = fallback

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)
        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = self.regex.search(resp)
                if match:
                    match = match.group(1).strip()
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered

        # print(resps)
        filtered_resps = list(map(lambda x: filter_set(x), resps))
        # print(filtered_resps)

        return filtered_resps


class WhitespaceFilter(Filter):
    """ """

    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def filter_set(inst):
            filtered_resp = []
            for resp in inst:
                if resp.startswith(" "):
                    resp = resp[1:]

                filtered_resp.append(resp)

            return filtered_resp

        filtered_resps = [filter_set(resp) for resp in resps]

        return filtered_resps


class ExtractJSONFilter(Filter):
    "Extract json from response. If no json is found, return an empty response."
    def __init__(self, default) -> None:
        self.default = default

    def _find_json(self, text):
        """
        Find and parse the first valid JSON appearing in a mixed text.
        """
        if isinstance(self.default, dict):
            start_delim, end_delim = ("{", "}")
        elif isinstance(self.default, list):
            start_delim, end_delim = ("[", "]")
        else:
            raise ValueError("Unexpected default type.")

        depth = 0
        start_index = -1

        for i, char in enumerate(text):
            if char == start_delim:
                if depth == 0:
                    start_index = i
                depth += 1
            elif char == end_delim:
                depth -= 1
                if depth == 0 and start_index != -1:
                    try:
                        json_obj = json.loads(text[start_index:i + 1])
                    except json.JSONDecodeError:
                        start_index = -1  # Reset the start index and keep going.
                    else:
                        return json_obj, "extract_success"

        # If we get to the end without finding anything, return default.
        return self.default, "extract_failure"

    def apply(self, resps, docs):
        # Keep track of the final status for each. We don't use this right now, but
        # could at some point.
        counts = Counter()

        def filter_set(inst):
            if len(inst) > 1:
                raise ValueError("Expected a single response.")
            resp = inst[0]
            try:
                filtered_resp = json.loads(resp)
            except json.JSONDecodeError:
                filtered_resp, status = self._find_json(resp)
            else:
                status = "valid"

            counts[status] += 1

            # If the keys are wrong, use the default.
            if isinstance(self.default, dict):
                if set(self.default.keys()) != set(filtered_resp.keys()):
                    filtered_resp = self.default
            
            # Convert back to string so it's the same type as gold.
            return json.dumps(filtered_resp)

        filtered_resps = [filter_set(resp) for resp in resps]

        return filtered_resps
