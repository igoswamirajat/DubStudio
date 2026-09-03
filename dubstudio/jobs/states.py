STATES = (
    "created",
    "queued",
    "ingesting",
    "extracting",
    "separating",
    "transcribing",
    "diarizing",
    "segmenting",
    "translating",
    "enrolling_voices",
    "synthesizing",
    "fitting",
    "mixing",
    "exporting",
    "completed",
    "failed",
    "canceled",
)

PIPELINE_STAGES = [
    "ingesting",
    "extracting",
    "separating",
    "transcribing",
    "diarizing",
    "segmenting",
    "translating",
    "enrolling_voices",
    "synthesizing",
    "fitting",
    "mixing",
    "exporting",
]

_FORWARD = {
    "created": {"queued"},
    "queued": {"ingesting"},
    "ingesting": {"extracting", "failed"},
    "extracting": {"separating", "failed"},
    "separating": {"transcribing", "failed"},
    "transcribing": {"diarizing", "failed"},
    "diarizing": {"segmenting", "failed"},
    "segmenting": {"translating", "failed"},
    "translating": {"enrolling_voices", "failed"},
    "enrolling_voices": {"synthesizing", "failed"},
    "synthesizing": {"fitting", "failed"},
    "fitting": {"mixing", "failed"},
    "mixing": {"exporting", "failed"},
    "exporting": {"completed", "failed"},
    "failed": {"queued"},
    "canceled": {"queued"},
}

ALLOWED: dict[str, set[str]] = {k: set(v) for k, v in _FORWARD.items()}

_order = ["queued"] + PIPELINE_STAGES + ["completed"]
for i, s in enumerate(_order):
    ALLOWED.setdefault(s, set())
    for later in _order[i + 1 :]:
        ALLOWED[s].add(later)
    ALLOWED[s].add("failed")

NON_TERMINAL = set(STATES) - {"completed", "failed", "canceled"}
for s in NON_TERMINAL:
    ALLOWED.setdefault(s, set()).add("canceled")


def can_transition(src: str, dst: str) -> bool:
    return dst in ALLOWED.get(src, set())


def require_transition(src: str, dst: str) -> None:
    if not can_transition(src, dst):
        raise ValueError(f"illegal transition {src} -> {dst}")
