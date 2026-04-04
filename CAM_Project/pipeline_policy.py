QUERY_EXCEPTION_TERMS = {"resolved", "remission", "history of", "past", "follow-up"}
CAUSAL_TRIGGER_TERMS = ("associated with", "due to", "caused by", "secondary to")
PREGNANCY_TRIGGER_TERMS = ("pregnancy", "childbirth", "pre-existing", "complicating pregnancy")

CORE_BLOCK_TERMS = (
    "benign",
    "labile",
    "malignant",
    "systolic",
    "diastolic",
    "renal",
    "with",
    "associated",
    "due to",
)

NARROW_SUBTYPE_TERMS = (
    "secondary",
    "renovascular",
    "drug-induced",
    "brittle",
    "goldblatt",
    "malignant",
    "juvenile",
    "gestational",
    "type 1",
    "type 2",
    "renal",
    "albuminuria",
    "systolic",
    "diastolic",
    "rebound",
    "benign",
    "supine",
    "central",
    "generalized",
    "simple",
    "localized",
    "childhood",
    "ulcer",
    "remission",
    "complication",
)

DEFAULT_CANDIDATE_POOL_LIMIT = 15
DEFAULT_INCLUDE_CANDIDATES_CAP = 4
DEFAULT_REVIEW_CANDIDATES_CAP = 3
DEFAULT_SPECIFIC_VARIANTS_CAP = 5

RETRIEVAL_HISTORY_EXCEPTION_TERMS = {"resolved", "remission", "history", "past", "follow", "inactive"}
RETRIEVAL_PREGNANCY_EXCEPTION_TERMS = {
    "pregnancy",
    "childbirth",
    "puerperium",
    "eclampsia",
    "pre",
    "partum",
}
RETRIEVAL_PREGNANCY_MARKERS = (
    "pregnancy",
    "childbirth",
    "puerperium",
    "eclampsia",
    "pre-eclampsia",
    "preeclampsia",
)
