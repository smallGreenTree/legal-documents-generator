"""Constants for controlled morphological augmentation."""

from src.synthetic_ner.types.augmentation import MorphologyTransformation

AUGMENTATION_DIRECTORY_NAME = "augmentations"
AUGMENTATION_MANIFEST_FILENAME = "augmentation_manifest.json"
MORPHOLOGY_BATCH_REPORT_FILENAME = "morphology_batch_report.json"
VARIANT_VERSION = 1
MORPHOLOGY_MODEL_STAGE = "morphology"
MORPHOLOGY_PIPELINE_STAGE = "morphological_augmentation"
MAX_CUSTOM_STYLE_CHARS = 200
MAX_STYLE_SLUG_CHARS = 48
MIN_STYLE_TEMPERATURE = 0.0
MAX_STYLE_TEMPERATURE = 1.5
FLAT_GROUNDTRUTH_PREFIX = "groundtruth_"
FLAT_GROUNDTRUTH_HEADER = (
    "doc_id",
    "entity_text",
    "label",
    "should_propose",
    "notes",
)

DETERMINISTIC_TRANSFORMATIONS = frozenset(
    {
        MorphologyTransformation.INTENTIONAL_TYPOS,
        MorphologyTransformation.RANDOM_LAYOUT,
    }
)
TYPO_EXCLUDED_WORDS = frozenset(
    {
        "against",
        "except",
        "excluding",
        "never",
        "neither",
        "nobody",
        "none",
        "nothing",
        "without",
    }
)

MENTION_TOKEN_TEMPLATE = "⟦NER_{index:04d}⟧"
LITERAL_TOKEN_TEMPLATE = "⟦LITERAL_{index:04d}⟧"
PROTECTED_TOKEN_PATTERN = r"⟦(?:NER|LITERAL)_\d{4}⟧"
NUMERIC_LITERAL_PATTERN = (
    r"(?<!\w)(?:[£€$]\s*)?\d+(?:[.,]\d+)*(?:[/:-]\d+)*(?:\([0-9A-Za-z]+\))*(?:%)?"
)

TRANSFORMATION_INSTRUCTIONS = {
    MorphologyTransformation.ACTIVE_TO_PASSIVE: (
        "Convert eligible active-voice clauses to natural passive-voice clauses. "
        "Keep the same actors, actions, objects, attribution, tense, and chronology."
    ),
    MorphologyTransformation.VERBAL_TO_NOMINAL: (
        "Convert eligible verbal clauses into natural nominal constructions. "
        "Keep the same actors, actions, objects, attribution, tense, and chronology."
    ),
    MorphologyTransformation.POSSESSIVE_REFRAME: (
        "Reframe eligible possessive and of-phrases between possessive and prepositional "
        "forms. Keep ownership, attribution, relationships, and chronology unchanged."
    ),
    MorphologyTransformation.INTENTIONAL_TYPOS: (
        "Introduce a small, reproducible set of internal letter transpositions in "
        "unannotated context words. Protected entities and numeric facts remain unchanged."
    ),
    MorphologyTransformation.RANDOM_LAYOUT: (
        "Apply reproducible indentation, line-wrapping, and paragraph-spacing variation "
        "without changing non-whitespace content or protected values."
    ),
}
